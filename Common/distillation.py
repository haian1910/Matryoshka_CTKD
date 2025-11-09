import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed
import shutil
import json
from tqdm import tqdm
import math
from transformers import AutoTokenizer
from transformers.integrations import HfDeepSpeedConfig
from Common.arguments import get_args
from Common.distiller import Distiller
from Common.data_utils.distill_datasets import DistillDataset
from Common.utils import (
    initialize,
    get_optimizer, 
    get_learning_rate_scheduler,
    print_rank, 
    log_rank,
    all_gather,
)
from Common.criterions import build_criterion
from Common.all_data_utils.sts_dataset import STSDataset
from Common.all_data_utils.nli_dataset import NLIDataset
from Common.all_data_utils.clf_dataset import CLFDataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_score, recall_score, accuracy_score

torch.set_num_threads(4)

def truncate_embeddings(embeddings, dim):
    """
    Truncate embeddings to specified dimension for MRL evaluation.
    
    Args:
        embeddings: torch.Tensor of shape (batch_size, hidden_size)
        dim: int, target dimension to truncate to
    
    Returns:
        Truncated embeddings of shape (batch_size, dim)
    """
    if dim >= embeddings.size(-1):
        return embeddings
    return embeddings[:, :dim]

class CLFHead(nn.Module):
    """Classification head that wraps a base model with a linear classifier"""
    def __init__(self, base_model, hidden_size, num_labels, dropout_rate=0.1):
        super(CLFHead, self).__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        # Add dropout and linear classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """Forward pass with frozen base model and trainable classifier"""
        # Get embeddings from frozen base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Mean pooling
        last_hidden = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # Apply dropout and classifier
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return type('Output', (), {
            'logits': logits,
            'loss': loss,
            'last_hidden_state': outputs.last_hidden_state
        })()

def prepare_dataset(args, distiller):
    """Prepare only training dataset"""
    data = {}
    if args.do_train:
        data["train"] = DistillDataset(
            args, 
            distiller.student_tokenizer,
            distiller.teacher_tokenizer
        )
        print("Num of train data: {}".format(len(data["train"])))
    else:
        raise ValueError("do_train must be set to True")
        
    return data

def prepare_sts_dataset(args, distiller):
    """Prepare STS dev/test datasets for evaluation after training"""
    sts_data = {}
    
    # Check if STS data directory exists
    sts_data_dir = getattr(args, 'sts_data_dir', None)
    if not sts_data_dir:
        print("Warning: sts_data_dir not specified, skipping STS evaluation")
        return sts_data
    
    if not os.path.exists(sts_data_dir):
        print(f"Warning: STS data directory {sts_data_dir} does not exist")
        return sts_data
    
    # Temporarily change data_dir to load STS data
    original_data_dir = args.data_dir
    args.data_dir = sts_data_dir
    
    try:
        dev_path = os.path.join(sts_data_dir, "dev.csv")
        if os.path.exists(dev_path):
            sts_data["dev"] = STSDataset(
                args,
                "dev",
                distiller.student_tokenizer,
                distiller.teacher_tokenizer
            )
            print("Num of STS dev data: {}".format(len(sts_data["dev"])))
        
        test_path = os.path.join(sts_data_dir, "test.csv")
        if os.path.exists(test_path):
            sts_data["test"] = STSDataset(
                args,
                "test",
                distiller.student_tokenizer,
                distiller.teacher_tokenizer
            )
            print("Num of STS test data: {}".format(len(sts_data["test"])))
    finally:
        # Restore original data_dir
        args.data_dir = original_data_dir
    
    return sts_data

def prepare_nli_dataset(args, distiller):
    """Prepare NLI dev/test datasets for evaluation after training"""
    nli_data = {}
    
    # Check if NLI data directory exists
    nli_data_dir = getattr(args, 'nli_data_dir', None)
    if not nli_data_dir:
        print("Warning: nli_data_dir not specified, skipping NLI evaluation")
        return nli_data
    
    if not os.path.exists(nli_data_dir):
        print(f"Warning: NLI data directory {nli_data_dir} does not exist")
        return nli_data
    
    # Temporarily change data_dir to load NLI data
    original_data_dir = args.data_dir
    args.data_dir = nli_data_dir
    
    try:
        dev_path = os.path.join(nli_data_dir, "dev.csv")
        if os.path.exists(dev_path):
            nli_data["dev"] = NLIDataset(
                args,
                "dev",
                distiller.student_tokenizer,
                distiller.teacher_tokenizer
            )
            print("Num of NLI dev data: {}".format(len(nli_data["dev"])))
        
        test_path = os.path.join(nli_data_dir, "test.csv")
        if os.path.exists(test_path):
            nli_data["test"] = NLIDataset(
                args,
                "test",
                distiller.student_tokenizer,
                distiller.teacher_tokenizer
            )
            print("Num of NLI test data: {}".format(len(nli_data["test"])))
    finally:
        # Restore original data_dir
        args.data_dir = original_data_dir
    
    return nli_data

def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device):
    print("Start Contrastive Learning Training")
    start_time = time.time()

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        criterion = build_criterion(args)

    sampler = DistributedSampler(
        dataset["train"], 
        shuffle=True, 
        drop_last=True, 
        rank=dp_rank, 
        num_replicas=dp_world_size
    )
    train_loader = DataLoader(
        dataset['train'], 
        sampler=sampler, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset["train"].collate
    )
    
    step = 0
    model_list = []
    logging_output = {
        "epoch": 0,
        "global_step": 0,
        "loss": [], 
        "correct": [],
        "micro_step_time": [],
        "step_time": []
    }
    
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        logging_output["epoch"] += 1
        
        print("Start iterations of epoch {}".format(epoch + 1))
        model.train()
        print("Training mode: {}".format(model.student_model.training))

        epoch_start_time = time.time()
        step = 0
        total_samples = 0
        total_time = 0.0

        data_iter = train_loader
        if dist.get_rank() == 0:
            data_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)

        for batch in data_iter:
            st_time = time.time()
            input_batch = batch  # No output_batch for contrastive learning
            dataset["train"].move_to_device(input_batch, device)

            loss, logging_output = model(
                criterion,
                {"input_batch": input_batch, "output_batch": None},
                logging_output,
                loss_denom=1,  # deepspeed supports sync gradient, no need to calculate loss_denom
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Loss is NaN or Inf. Skipping this step.")
                continue

            model.backward(loss)
            model.step()
            torch.cuda.synchronize()  # correctly compute time

            elapsed_time = time.time() - st_time
            num_samples = input_batch["query_input_ids"].size(0)
            total_samples += num_samples
            total_time += elapsed_time
            step += 1

            logging_output["global_step"] += 1
            logging_output["micro_step_time"].append(elapsed_time)
            logging_output["step_time"].append(elapsed_time)

            if dist.get_rank() == 0:
                # Calculate average accuracy for this epoch
                avg_accuracy = sum(logging_output["correct"]) / len(logging_output["correct"]) if logging_output["correct"] else 0.0
                data_iter.set_postfix(
                    loss=loss.item(),
                    accuracy=avg_accuracy
                )

        # Log epoch statistics
        if dist.get_rank() == 0:
            epoch_time = time.time() - epoch_start_time
            avg_loss = sum(logging_output["loss"][-step:]) / step if logging_output["loss"] else 0.0
            total_correct = sum(logging_output["correct"][-step:])
            
            # Calculate total samples processed in this epoch across all GPUs
            total_samples_global = total_samples * dp_world_size
            avg_accuracy = total_correct / total_samples_global if total_samples_global > 0 else 0.0
            
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average Accuracy: {avg_accuracy:.4f}")
            print(f"  Samples Processed: {total_samples_global}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Throughput: {total_samples_global / epoch_time:.2f} samples/s")

        # Save checkpoint at intervals
        if dist.get_rank() == 0 and args.save_dir and (epoch + 1) % args.save_interval == 0:
            # Calculate metrics for checkpoint naming
            recent_loss = sum(logging_output["loss"][-step:]) / step if logging_output["loss"] else 0.0
            recent_correct = sum(logging_output["correct"][-step:])
            recent_accuracy = recent_correct / (total_samples * dp_world_size) if total_samples > 0 else 0.0
            
            ckpt_name = "epoch{}_step{}_loss{:.4f}_acc{:.4f}".format(
                epoch + 1, 
                logging_output["global_step"], 
                recent_loss,
                recent_accuracy
            )
            save_dir_path = os.path.join(args.save_dir, ckpt_name)
            
            os.makedirs(save_dir_path, exist_ok=True)
            
            if not args.only_save_projector:
                print("Saving tokenizer...")
                tokenizer.save_pretrained(save_dir_path)
                print("Saving model...")
                model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
                print("Saving config...")
                model.module.student_model.config.save_pretrained(save_dir_path)
            
            if hasattr(model.module, "projectors"):
                print("Saving projector...")
                torch.save(
                    model.module.projectors.state_dict(), 
                    os.path.join(save_dir_path, "projector.pt")
                )
            
            # Keep track of saved models based on accuracy
            model_list.append({"path": save_dir_path, "score": recent_accuracy})
            model_list = sorted(model_list, key=lambda x: x["score"], reverse=True)  # Higher accuracy is better
            
            # Remove worst checkpoints if exceeding limit
            if len(model_list) > args.keep_best_n_checkpoints:
                removed_model = model_list.pop(-1)  # Remove lowest accuracy
                shutil.rmtree(removed_model["path"])
                print(f"Removed checkpoint: {removed_model['path']}")

            print(f"Model has been saved to {save_dir_path}")
        
        dist.barrier()
            
    total_seconds = time.time() - start_time
    print("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))
    
    # Save final model
    if dist.get_rank() == 0 and args.save_dir:
        final_save_dir = os.path.join(args.save_dir, "final_model")
        os.makedirs(final_save_dir, exist_ok=True)
        
        print("Saving final model...")
        tokenizer.save_pretrained(final_save_dir)
        model.module.student_model.save_pretrained(final_save_dir, safe_serialization=False)
        model.module.student_model.config.save_pretrained(final_save_dir)
        
        if hasattr(model.module, "projectors"):
            torch.save(
                model.module.projectors.state_dict(), 
                os.path.join(final_save_dir, "projector.pt")
            )
        
        print(f"Final model saved to {final_save_dir}")


@torch.no_grad()
def evaluate_sts(args, tokenizer, student_model, dataset, split, device):
    """
    Evaluate model on STS tasks with Pearson and Spearman correlations.
    Supports MRL evaluation at multiple dimensions.
    """
    if dist.get_rank() != 0:
        return None
    
    # Get matryoshka dimensions from args
    matryoshka_dims = getattr(args, 'matryoshka_dims', [16, 32, 64, 128, 256, 512, 768])  # Default to full dimension
    
    # Use regular DataLoader without DistributedSampler
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    
    # Store predictions and targets for each dimension
    dim_results = {dim: {"preds": [], "targets": []} for dim in matryoshka_dims}
    
    for input_batch, output_batch in tqdm(dataloader, desc=f"Evaluating STS {split}"):
        dataset.move_to_device([input_batch, output_batch], device)
        targets = output_batch["labels"]
        
        outputs = student_model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            token_type_ids=input_batch.get("token_type_ids", None)
        )
        
        # Get embeddings using mean pooling
        last_hidden = outputs.last_hidden_state
        attention_mask_expanded = input_batch["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # Ensure targets have compatible shape
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(-1)
        
        # Evaluate at each dimension
        for dim in matryoshka_dims:
            # Truncate embeddings to current dimension
            truncated_emb = truncate_embeddings(embeddings, dim)
            
            # Use cosine similarity as predictions (scaled to [0, 5])
            # For sentence pairs, we would compare two embeddings
            # Here we use embedding norm as a proxy (adjust based on your task)
            predictions = torch.norm(truncated_emb, dim=-1, keepdim=True)
            predictions = (predictions / predictions.max()) * 5.0
            
            dim_results[dim]["preds"].append(predictions)
            dim_results[dim]["targets"].append(targets)
    
    # Compute metrics for each dimension
    results = {}
    for dim in matryoshka_dims:
        all_preds = torch.cat(dim_results[dim]["preds"], dim=0)
        all_targets = torch.cat(dim_results[dim]["targets"], dim=0)
        
        # Convert to float32 before numpy conversion
        all_preds = all_preds.to(torch.float32).cpu().numpy().flatten()
        all_targets = all_targets.to(torch.float32).cpu().numpy().flatten()
        
        # Calculate correlations
        pearson_corr, _ = pearsonr(all_preds, all_targets)
        spearman_corr, _ = spearmanr(all_preds, all_targets)
        mse = ((all_preds - all_targets) ** 2).mean()
        
        results[dim] = {
            "pearson": round(float(pearson_corr), 6),
            "spearman": round(float(spearman_corr), 6),
            "mse": round(float(mse), 6)
        }
        
        print(f"STS {split} | Dim {dim:4d} | Pearson: {results[dim]['pearson']:.4f} | "
              f"Spearman: {results[dim]['spearman']:.4f} | MSE: {results[dim]['mse']:.4f}")
    
    student_model.train()
    return results

@torch.no_grad()
def evaluate_nli(args, tokenizer, student_model, dataset, split, device):
    """
    Evaluate model on NLI tasks with accuracy, precision, and recall.
    Supports MRL evaluation at multiple dimensions.
    """
    if dist.get_rank() != 0:
        return None
    
    # Get matryoshka dimensions from args
    matryoshka_dims = getattr(args, 'matryoshka_dims', [16, 32, 64, 128, 256, 512, 768])
    num_labels = getattr(args, 'num_labels', 3)  # Typically 3 for NLI
    
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    
    # Create classification heads for each dimension (simple linear layer)
    clf_heads = {}
    for dim in matryoshka_dims:
        clf_head = nn.Linear(dim, num_labels).to(device)
        # Initialize with xavier uniform
        nn.init.xavier_uniform_(clf_head.weight)
        nn.init.zeros_(clf_head.bias)
        clf_heads[dim] = clf_head
    
    # Store predictions and labels for each dimension
    dim_results = {dim: {"preds": [], "labels": [], "losses": []} for dim in matryoshka_dims}
    
    for input_batch, output_batch in tqdm(dataloader, desc=f"Evaluating NLI {split}"):
        dataset.move_to_device([input_batch, output_batch], device)
        labels = output_batch["labels"]
        
        outputs = student_model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            token_type_ids=input_batch.get("token_type_ids", None)
        )
        
        # Get embeddings using mean pooling
        last_hidden = outputs.last_hidden_state
        attention_mask_expanded = input_batch["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # Evaluate at each dimension
        for dim in matryoshka_dims:
            truncated_emb = truncate_embeddings(embeddings, dim)
            
            # Get logits from classification head
            logits = clf_heads[dim](truncated_emb)
            preds = logits.argmax(dim=-1)
            
            # Compute loss
            loss = F.cross_entropy(logits, labels)
            
            dim_results[dim]["preds"].append(preds)
            dim_results[dim]["labels"].append(labels)
            dim_results[dim]["losses"].append(loss.item() * labels.size(0))
    
    # Compute metrics for each dimension
    results = {}
    for dim in matryoshka_dims:
        all_preds = torch.cat(dim_results[dim]["preds"], dim=0).cpu().numpy()
        all_labels = torch.cat(dim_results[dim]["labels"], dim=0).cpu().numpy()
        total_loss = sum(dim_results[dim]["losses"])
        total_samples = len(all_labels)
        
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        results[dim] = {
            "accuracy": round(float(accuracy), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "loss": round(float(avg_loss), 6)
        }
        
        print(f"NLI {split} | Dim {dim:4d} | Acc: {results[dim]['accuracy']:.4f} | "
              f"Prec: {results[dim]['precision']:.4f} | Rec: {results[dim]['recall']:.4f} | "
              f"Loss: {results[dim]['loss']:.4f}")
    
    student_model.train()
    return results

@torch.no_grad()
def evaluate_clf(args, tokenizer, student_model, dataset, split, device):
    """
    Evaluate model on classification tasks with accuracy, precision, and recall.
    Supports MRL evaluation at multiple dimensions.
    """
    if dist.get_rank() != 0:
        return None
    
    # Get matryoshka dimensions from args
    matryoshka_dims = getattr(args, 'matryoshka_dims', [16, 32, 64, 128, 256, 512, 768])
    num_labels = getattr(args, 'num_labels', 2)  # Binary classification default
    
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    
    # Create classification heads for each dimension
    clf_heads = {}
    for dim in matryoshka_dims:
        clf_head = nn.Linear(dim, num_labels).to(device)
        nn.init.xavier_uniform_(clf_head.weight)
        nn.init.zeros_(clf_head.bias)
        clf_heads[dim] = clf_head
    
    # Store predictions and labels for each dimension
    dim_results = {dim: {"preds": [], "labels": [], "losses": []} for dim in matryoshka_dims}
    
    for input_batch, output_batch in tqdm(dataloader, desc=f"Evaluating CLF {split}"):
        dataset.move_to_device([input_batch, output_batch], device)
        labels = output_batch["labels"]
        
        outputs = student_model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            token_type_ids=input_batch.get("token_type_ids", None)
        )
        
        # Get embeddings using mean pooling
        last_hidden = outputs.last_hidden_state
        attention_mask_expanded = input_batch["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # Evaluate at each dimension
        for dim in matryoshka_dims:
            truncated_emb = truncate_embeddings(embeddings, dim)
            
            # Get logits from classification head
            logits = clf_heads[dim](truncated_emb)
            preds = logits.argmax(dim=-1)
            
            # Compute loss
            loss = F.cross_entropy(logits, labels)
            
            dim_results[dim]["preds"].append(preds)
            dim_results[dim]["labels"].append(labels)
            dim_results[dim]["losses"].append(loss.item() * labels.size(0))
    
    # Compute metrics for each dimension
    results = {}
    for dim in matryoshka_dims:
        all_preds = torch.cat(dim_results[dim]["preds"], dim=0).cpu().numpy()
        all_labels = torch.cat(dim_results[dim]["labels"], dim=0).cpu().numpy()
        total_loss = sum(dim_results[dim]["losses"])
        total_samples = len(all_labels)
        
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        results[dim] = {
            "accuracy": round(float(accuracy), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "loss": round(float(avg_loss), 6)
        }
        
        print(f"CLF {split} | Dim {dim:4d} | Acc: {results[dim]['accuracy']:.4f} | "
              f"Prec: {results[dim]['precision']:.4f} | Rec: {results[dim]['recall']:.4f} | "
              f"Loss: {results[dim]['loss']:.4f}")
    
    student_model.train()
    return results


def prepare_clf_dataset(args, distiller):
    """Prepare CLF train/dev datasets for classifier fine-tuning"""
    clf_data = {}
    
    # Check if CLF data directory exists
    clf_data_dir = getattr(args, 'clf_data_dir', None)
    if not clf_data_dir:
        print("Warning: clf_data_dir not specified, skipping CLF fine-tuning")
        return clf_data
    
    if not os.path.exists(clf_data_dir):
        print(f"Warning: CLF data directory {clf_data_dir} does not exist")
        return clf_data
    
    # Temporarily change data_dir to load CLF data
    original_data_dir = args.data_dir
    args.data_dir = clf_data_dir
    
    try:
        # Load train data
        train_path = os.path.join(clf_data_dir, "train.csv")
        if os.path.exists(train_path):
            clf_data["train"] = CLFDataset(
                args,
                "train",
                distiller.student_tokenizer,
                distiller.teacher_tokenizer
            )
            print("Num of CLF train data: {}".format(len(clf_data["train"])))
        
        # Load dev data
        dev_path = os.path.join(clf_data_dir, "dev.csv")
        if os.path.exists(dev_path):
            clf_data["dev"] = CLFDataset(
                args,
                "dev",
                distiller.student_tokenizer,
                distiller.teacher_tokenizer
            )
            print("Num of CLF dev data: {}".format(len(clf_data["dev"])))
    finally:
        # Restore original data_dir
        args.data_dir = original_data_dir
    
    return clf_data


def finetune_clf(args, tokenizer, model_engine, dataset, device):
    """Fine-tune classifier head on CLF dataset"""
    print("\n" + "="*50)
    print("Start Classification Head Fine-tuning (frozen base model)")
    print("="*50)
    start_time = time.time()

    if "train" not in dataset or "dev" not in dataset:
        print("Warning: train or dev dataset missing for CLF fine-tuning")
        return
    
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()

    sampler = DistributedSampler(
        dataset["train"], 
        shuffle=True, 
        drop_last=True, 
        rank=dp_rank, 
        num_replicas=dp_world_size
    )
    train_loader = DataLoader(
        dataset['train'], 
        sampler=sampler, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset["train"].collate
    )
    
    step = 0
    logging_output = {
        "epoch": 0,
        "global_step": 0,
        "loss": [], 
        "correct": [],
    }
    
    # CLF fine-tuning epochs (typically 3-5)
    clf_epochs = 3
    
    for epoch in range(clf_epochs):
        sampler.set_epoch(epoch)
        logging_output["epoch"] += 1
        
        print(f"\nCLF Fine-tuning Epoch {epoch + 1}/{clf_epochs}")
        model_engine.train()

        epoch_start_time = time.time()
        step = 0
        total_samples = 0

        data_iter = train_loader
        if dist.get_rank() == 0:
            data_iter = tqdm(train_loader, desc=f"CLF Epoch {epoch + 1}", dynamic_ncols=True)

        for batch in data_iter:
            st_time = time.time()
            input_batch, output_batch = batch
            dataset["train"].move_to_device([input_batch, output_batch], device)
            
            labels = output_batch["labels"]
            outputs = model_engine.module(
                input_ids=input_batch["input_ids"],
                attention_mask=input_batch["attention_mask"],
                token_type_ids=input_batch.get("token_type_ids", None),
                labels=labels
            )
            
            loss = outputs.loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Loss is NaN or Inf. Skipping this step.")
                continue

            model_engine.backward(loss)
            model_engine.step()
            torch.cuda.synchronize()

            num_samples = input_batch["input_ids"].size(0)
            total_samples += num_samples
            step += 1

            logging_output["global_step"] += 1
            logging_output["loss"].append(loss.item())
            
            # Calculate accuracy
            preds = outputs.logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            logging_output["correct"].append(correct)
            
            if dist.get_rank() == 0:
                avg_loss = sum(logging_output["loss"][-min(10, len(logging_output["loss"])):]) / min(10, len(logging_output["loss"]))
                avg_accuracy = sum(logging_output["correct"][-min(10, len(logging_output["correct"])):]) / min(10, len(logging_output["correct"]))
                data_iter.set_postfix(
                    loss=avg_loss,
                    accuracy=avg_accuracy
                )

        # Log epoch statistics
        if dist.get_rank() == 0:
            epoch_time = time.time() - epoch_start_time
            avg_loss = sum(logging_output["loss"][-step:]) / step if logging_output["loss"] else 0.0
            total_correct = sum(logging_output["correct"][-step:])
            
            total_samples_global = total_samples * dp_world_size
            avg_accuracy = total_correct / total_samples_global if total_samples_global > 0 else 0.0
            
            print(f"CLF Epoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average Accuracy: {avg_accuracy:.4f}")
            print(f"  Samples: {total_samples_global}")
            print(f"  Time: {epoch_time:.2f}s")
        
        dist.barrier()
    
    total_seconds = time.time() - start_time
    print("\nClassification Fine-tuning Done in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))
    
    # Evaluate on CLF dev set
    print("\nEvaluating on CLF dev set...")
    model_engine.eval()
    dev_loss, dev_accuracy, dev_precision, dev_recall = evaluate_clf_dev(
        args, 
        model_engine.module,
        dataset["dev"],
        device
    )
    print(f"CLF Dev Results - Loss: {dev_loss}, Acc: {dev_accuracy}, Prec: {dev_precision}, Rec: {dev_recall}")


@torch.no_grad()
def evaluate_clf_dev(args, model, dataset, device):
    """Evaluate classifier on dev set"""
    if dist.get_rank() != 0:
        return None, None, None, None        
    
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_samples = 0
    
    for input_batch, output_batch in tqdm(dataloader, desc="CLF Evaluation"):
        dataset.move_to_device([input_batch, output_batch], device)
        labels = output_batch["labels"]
        
        outputs = model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            token_type_ids=input_batch.get("token_type_ids", None),
            labels=labels
        )
        
        logits = outputs.logits
        loss = outputs.loss
        preds = logits.argmax(dim=-1)
        
        all_preds.append(preds)
        all_labels.append(labels)
        sample_num = labels.size(0)
        total_loss += loss.item() * sample_num
        total_samples += sample_num

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    precision = round(float(precision), 6)
    recall = round(float(recall), 6)
    accuracy = round(float(accuracy), 6)
    avg_loss = round(float(avg_loss), 6)

    return avg_loss, accuracy, precision, recall

def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)
    dp_world_size = dist.get_world_size()

    # Save arguments
    if dist.get_rank() == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    device = torch.cuda.current_device()

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_rank("\n\n" + "="*30 + f" Contrastive Learning EXP at {cur_time} " + "="*30)
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    print('DeepSpeed config: {}'.format(ds_config))

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * dp_world_size

    print("Initializing a distiller for contrastive learning...")
    distiller = Distiller(args, device)
    dataset = prepare_dataset(args, distiller)
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size))
        assert args.total_iters is not None or args.num_epochs is not None
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.num_epochs
        if args.num_epochs is None:
            args.num_epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)

        print("Total iterations: {}".format(args.total_iters))
        print("Number of epochs: {}".format(args.num_epochs))
        print("Iterations per epoch: {}".format(args.train_iters_per_epoch))
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    optimizer_grouped_parameters = get_optimizer(args, distiller.student_model)
    # No projectors for contrastive learning on embeddings only
    # optimizer_grouped_parameters = distiller.add_optimizer_param_group(optimizer_grouped_parameters)

    lr_scheduler = get_learning_rate_scheduler(args, optimizer_grouped_parameters)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=distiller,
        optimizer=optimizer_grouped_parameters,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    if args.do_train:
        finetune(args, distiller.student_tokenizer, model_engine, optimizer, lr_scheduler, dataset, device)
    else:
        raise ValueError("do_train must be set to True for contrastive learning")
    
    # Replace the STS evaluation section in main() with:
    if args.do_train and hasattr(args, 'do_eval') and args.do_eval:
        print("Preparing STS evaluation datasets...")
        sts_eval_data = prepare_sts_dataset(args, distiller)
        
        if "dev" in sts_eval_data:
            print("\n" + "="*60)
            print("Evaluating on STS dev set with MRL dimensions")
            print("="*60)
            dev_results = evaluate_sts(
                args, 
                distiller.student_tokenizer, 
                model_engine.module.student_model, 
                sts_eval_data["dev"], 
                "dev", 
                device
            )
        
        if "test" in sts_eval_data:
            print("\n" + "="*60)
            print("Evaluating on STS test set with MRL dimensions")
            print("="*60)
            test_results = evaluate_sts(
                args, 
                distiller.student_tokenizer, 
                model_engine.module.student_model, 
                sts_eval_data["test"], 
                "test", 
                device
            )
        
    
    # # Evaluate on NLI dev/test sets after training if do_eval is enabled
    # if args.do_train and hasattr(args, 'do_eval') and args.do_eval:
    #     print("Preparing NLI evaluation datasets...")
    #     nli_eval_data = prepare_nli_dataset(args, distiller)
        
    #     if "dev" in nli_eval_data:
    #         print("Evaluating on NLI dev set...")
    #         dev_loss, dev_accuracy, dev_precision, dev_recall = evaluate_nli(
    #             args, 
    #             distiller.student_tokenizer, 
    #             model_engine.module.student_model, 
    #             nli_eval_data["dev"], 
    #             "dev", 
    #             device
    #         )
        
    #     if "test" in nli_eval_data:
    #         print("Evaluating on NLI test set...")
    #         test_loss, test_accuracy, test_precision, test_recall = evaluate_nli(
    #             args, 
    #             distiller.student_tokenizer, 
    #             model_engine.module.student_model, 
    #             nli_eval_data["test"], 
    #             "test", 
    #             device
    #         )
    
    # Evaluate on CLF dev/test sets after training if do_eval is enabled
    print("\n" + "="*60)
    print("Checking for CLF fine-tuning...")
    print("="*60)
    clf_data = prepare_clf_dataset(args, distiller)
    
    if clf_data and "train" in clf_data and "dev" in clf_data:
        # Create classification head
        print("Creating classification head with frozen base model...")
        hidden_size = distiller.student_model.config.hidden_size
        num_labels = args.num_labels
        
        clf_model = CLFHead(
            model_engine.module.student_model,
            hidden_size,
            num_labels,
            dropout_rate=0.1
        )
        
        # Optimize only classifier parameters
        clf_optimizer_params = [
            {
                "params": clf_model.classifier.parameters(),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": clf_model.dropout.parameters(),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
        ]
        
        # Create optimizer for classifier
        clf_optimizer = AdamW(clf_optimizer_params)
        
        # Simple deepspeed config for classifier fine-tuning
        clf_ds_config = {
            "train_batch_size": args.batch_size * args.gradient_accumulation_steps * dp_world_size,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {"type": "AdamW", "params": {"lr": args.lr}},
            "fp16": {"enabled": False},
            "bf16": {"enabled": True} if "bf16" in str(args.model_dtype) else {"enabled": False}
        }
        
        clf_engine, clf_optimizer, _, _ = deepspeed.initialize(
            model=clf_model,
            optimizer=clf_optimizer,
            config_params=clf_ds_config
        )
        
        # Fine-tune classifier
        finetune_clf(args, distiller.student_tokenizer, clf_engine, clf_data, device)
        
        print("Classification fine-tuning complete!")
    else:
        print("Skipping CLF fine-tuning (clf_data_dir not specified or missing train/dev data)")
    
    # ========== CLF Evaluation ==========
    print("\n[3/3] Preparing CLF evaluation datasets...")
    clf_eval_data = prepare_clf_dataset(args, distiller)
    
    if "dev" in clf_eval_data:
        print("\n" + "="*60)
        print("Evaluating on CLF dev set with MRL dimensions")
        print("="*60)
        clf_dev_results = evaluate_clf(
            args, 
            distiller.student_tokenizer, 
            model_engine.module.student_model, 
            clf_eval_data["dev"], 
            "dev", 
            device
        )
    
    if "test" in clf_eval_data:
        print("\n" + "="*60)
        print("Evaluating on CLF test set with MRL dimensions")
        print("="*60)
        clf_test_results = evaluate_clf(
            args, 
            distiller.student_tokenizer, 
            model_engine.module.student_model, 
            clf_eval_data["test"], 
            "test", 
            device
        )
    
    print("\n" + "="*80)
    print(" EVALUATION COMPLETE ".center(80, "="))
    print("="*80)
    
if __name__ == "__main__":
    main()
