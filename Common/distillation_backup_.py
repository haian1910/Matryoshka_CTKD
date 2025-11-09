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


def prepare_sts_datasets_list(args, distiller):
    """Prepare STS datasets from a list of directories for multi-dataset evaluation"""
    sts_datasets = {}
    
    # Get list of STS data directories
    sts_data_dirs = getattr(args, 'sts_data_dirs', [])
    if not sts_data_dirs:
        print("Warning: sts_data_dirs not specified, skipping STS evaluation")
        return sts_datasets
    
    # Ensure it's a list
    if isinstance(sts_data_dirs, str):
        sts_data_dirs = [sts_data_dirs]
    
    original_data_dir = args.data_dir
    
    for sts_data_dir in sts_data_dirs:
        if not os.path.exists(sts_data_dir):
            print(f"Warning: STS data directory {sts_data_dir} does not exist, skipping")
            continue
        
        dataset_name = os.path.basename(sts_data_dir.rstrip('/'))
        args.data_dir = sts_data_dir
        
        try:
            dataset_dict = {}
            
            dev_path = os.path.join(sts_data_dir, "dev.csv")
            if os.path.exists(dev_path):
                dataset_dict["dev"] = STSDataset(
                    args,
                    "dev",
                    distiller.student_tokenizer,
                    distiller.teacher_tokenizer
                )
                print(f"  Loaded STS '{dataset_name}' dev: {len(dataset_dict['dev'])} samples")
            
            test_path = os.path.join(sts_data_dir, "test.csv")
            if os.path.exists(test_path):
                dataset_dict["test"] = STSDataset(
                    args,
                    "test",
                    distiller.student_tokenizer,
                    distiller.teacher_tokenizer
                )
                print(f"  Loaded STS '{dataset_name}' test: {len(dataset_dict['test'])} samples")
            
            if dataset_dict:
                sts_datasets[dataset_name] = dataset_dict
        
        except Exception as e:
            print(f"Error loading STS dataset from {sts_data_dir}: {e}")
            continue
    
    args.data_dir = original_data_dir
    return sts_datasets

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
    
    For each sentence pair:
    1. Extract embedding for sentence1
    2. Extract embedding for sentence2
    3. Compute cosine similarity between embeddings
    4. Compare with ground truth similarity scores
    """
    if dist.get_rank() != 0:
        return None
    
    # Get matryoshka dimensions from args
    matryoshka_dims = getattr(args, 'matryoshka_dims', [16, 32, 64, 128, 256, 512, 768])
    
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
        
        # Extract embeddings for sentence 1
        outputs_sent1 = student_model(
            input_ids=input_batch["input_ids_sent1"],
            attention_mask=input_batch["attention_mask_sent1"],
            token_type_ids=input_batch.get("token_type_ids_sent1", None)
        )
        
        # Extract embeddings for sentence 2
        outputs_sent2 = student_model(
            input_ids=input_batch["input_ids_sent2"],
            attention_mask=input_batch["attention_mask_sent2"],
            token_type_ids=input_batch.get("token_type_ids_sent2", None)
        )
        
        # Get embeddings using mean pooling for sentence 1
        last_hidden_sent1 = outputs_sent1.last_hidden_state
        attention_mask_expanded_sent1 = input_batch["attention_mask_sent1"].unsqueeze(-1).expand(last_hidden_sent1.size()).float()
        sum_embeddings_sent1 = torch.sum(last_hidden_sent1 * attention_mask_expanded_sent1, 1)
        sum_mask_sent1 = torch.clamp(attention_mask_expanded_sent1.sum(1), min=1e-9)
        embeddings_sent1 = sum_embeddings_sent1 / sum_mask_sent1
        
        # Get embeddings using mean pooling for sentence 2
        last_hidden_sent2 = outputs_sent2.last_hidden_state
        attention_mask_expanded_sent2 = input_batch["attention_mask_sent2"].unsqueeze(-1).expand(last_hidden_sent2.size()).float()
        sum_embeddings_sent2 = torch.sum(last_hidden_sent2 * attention_mask_expanded_sent2, 1)
        sum_mask_sent2 = torch.clamp(attention_mask_expanded_sent2.sum(1), min=1e-9)
        embeddings_sent2 = sum_embeddings_sent2 / sum_mask_sent2
        
        # Ensure targets have compatible shape
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(-1)
        
        # Evaluate at each dimension
        for dim in matryoshka_dims:
            # Truncate embeddings to current dimension
            truncated_emb_sent1 = truncate_embeddings(embeddings_sent1, dim)
            truncated_emb_sent2 = truncate_embeddings(embeddings_sent2, dim)
            
            # Compute cosine similarity between the two embeddings
            # cosine_similarity returns values in [-1, 1], scale to [0, 5]
            cosine_sim = F.cosine_similarity(truncated_emb_sent1, truncated_emb_sent2, dim=1)
            predictions = (cosine_sim + 1) / 2 * 5.0  # Scale from [-1, 1] to [0, 5]
            predictions = predictions.unsqueeze(-1)
            
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


def prepare_clf_datasets_list(args, distiller):
    """Prepare CLF datasets from a list of directories for multi-dataset evaluation"""
    clf_datasets = {}
    
    # Get list of CLF data directories
    clf_data_dirs = getattr(args, 'clf_data_dirs', [])
    if not clf_data_dirs:
        print("Warning: clf_data_dirs not specified, skipping CLF evaluation")
        return clf_datasets
    
    # Ensure it's a list
    if isinstance(clf_data_dirs, str):
        clf_data_dirs = [clf_data_dirs]
    
    original_data_dir = args.data_dir
    
    for clf_data_dir in clf_data_dirs:
        if not os.path.exists(clf_data_dir):
            print(f"Warning: CLF data directory {clf_data_dir} does not exist, skipping")
            continue
        
        dataset_name = os.path.basename(clf_data_dir.rstrip('/'))
        args.data_dir = clf_data_dir
        
        try:
            dataset_dict = {}
            
            # Load train data
            train_path = os.path.join(clf_data_dir, "train.csv")
            if os.path.exists(train_path):
                dataset_dict["train"] = CLFDataset(
                    args,
                    "train",
                    distiller.student_tokenizer,
                    distiller.teacher_tokenizer
                )
                print(f"  Loaded CLF '{dataset_name}' train: {len(dataset_dict['train'])} samples")
            
            # Load dev data
            dev_path = os.path.join(clf_data_dir, "dev.csv")
            if os.path.exists(dev_path):
                dataset_dict["dev"] = CLFDataset(
                    args,
                    "dev",
                    distiller.student_tokenizer,
                    distiller.teacher_tokenizer
                )
                print(f"  Loaded CLF '{dataset_name}' dev: {len(dataset_dict['dev'])} samples")
            
            if dataset_dict:
                clf_datasets[dataset_name] = dataset_dict
        
        except Exception as e:
            print(f"Error loading CLF dataset from {clf_data_dir}: {e}")
            continue
    
    args.data_dir = original_data_dir
    return clf_datasets


@torch.no_grad()
def extract_embeddings_efficient(model, dataloader, dataset, device):
    """
    Efficiently extract embeddings with mean pooling.
    Returns embeddings and labels in CPU memory to save GPU memory.
    """
    embeddings_list = []
    labels_list = []
    
    for input_batch, output_batch in tqdm(dataloader, desc="Extracting embeddings"):
        dataset.move_to_device([input_batch, output_batch], device)
        labels = output_batch["labels"]
        
        outputs = model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            token_type_ids=input_batch.get("token_type_ids", None)
        )
        
        # Mean pooling
        last_hidden = outputs.last_hidden_state
        attention_mask_expanded = input_batch["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # Move to CPU to save GPU memory
        embeddings_list.append(embeddings.cpu())
        labels_list.append(labels.cpu())
    
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return embeddings, labels


@torch.no_grad()
def evaluate_clf_batch_efficient(model, embeddings_full, labels, dim, num_labels, device, eval_batch_size=256):
    """
    Efficiently evaluate model in batches to avoid OOM on large datasets.
    Handles dimension truncation on-the-fly during evaluation.
    """
    all_preds = []
    all_logits = []
    total_loss = 0
    
    # Truncate embeddings to target dimension
    embeddings_truncated = truncate_embeddings(embeddings_full, dim)
    labels = labels.to(device)
    
    # Evaluate in batches
    for i in range(0, len(embeddings_truncated), eval_batch_size):
        batch_emb = embeddings_truncated[i:i+eval_batch_size].to(device)
        batch_lbl = labels[i:i+eval_batch_size]
        
        logits = model(batch_emb)
        loss = F.cross_entropy(logits, batch_lbl)
        
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_logits.append(logits.cpu())
        total_loss += loss.item() * len(batch_lbl)
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    labels_np = labels.cpu().numpy()
    avg_loss = total_loss / len(labels)
    
    accuracy = accuracy_score(labels_np, all_preds)
    precision = precision_score(labels_np, all_preds, average='macro', zero_division=0)
    recall = recall_score(labels_np, all_preds, average='macro', zero_division=0)
    
    return accuracy, precision, recall, avg_loss


def train_clf_head_batch_efficient(clf_head, train_emb_full, train_labels, dim, num_labels, 
                                   optimizer, clf_epochs, batch_size, device):
    """
    Train linear classifier efficiently in batches, handling dimension truncation.
    Embeddings must require gradients through the linear head.
    """
    # Truncate embeddings to target dimension once
    train_emb = truncate_embeddings(train_emb_full, dim)
    train_lbl = train_labels.to(device)
    
    for epoch in range(clf_epochs):
        clf_head.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # Shuffle indices
        indices = torch.randperm(len(train_emb))
        
        for i in range(0, len(train_emb), batch_size):
            batch_indices = indices[i:i+batch_size]
            # Move embeddings to device and ensure they require grad for backprop
            batch_emb = train_emb[batch_indices].to(device).requires_grad_(True)
            batch_lbl = train_lbl[batch_indices]
            
            optimizer.zero_grad()
            logits = clf_head(batch_emb)
            loss = F.cross_entropy(logits, batch_lbl)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_lbl)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == batch_lbl).sum().item()
            total_samples += len(batch_lbl)
            
            # Clear GPU cache periodically
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f"    Epoch {epoch+1}/{clf_epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")


def finetune_clf_per_dimension(args, tokenizer, student_model, dataset, device):
    """
    Efficient per-dimension classification fine-tuning for large datasets.
    
    Optimization strategy:
    1. Extract embeddings once (keeps full dimension on CPU)
    2. For each dimension:
       - Truncate embeddings on-demand
       - Train linear head in batches
       - Evaluate in batches
       - Prefer test set if available, otherwise use dev set
    """
    print("\n" + "="*50)
    print("Start Per-Dimension Classification Fine-tuning")
    print("="*50)
    
    if "train" not in dataset or "dev" not in dataset:
        print("Warning: train or dev dataset missing for CLF fine-tuning")
        return {}
    
    matryoshka_dims = getattr(args, 'matryoshka_dims', [16, 32, 64, 128, 256, 512, 768])
    num_labels = getattr(args, 'num_labels', 2)
    clf_epochs = getattr(args, 'clf_epochs', 3)
    eval_batch_size = getattr(args, 'eval_batch_size', 256)
    
    # Results for all dimensions
    all_dim_results = {}
    
    # ========== STEP 1: Extract embeddings from frozen base model ==========
    print("\n" + "="*60)
    print("Step 1: Extracting embeddings from frozen base model...")
    print("="*60)
    
    # Freeze the base model - keep it in eval mode
    student_model.eval()
    for param in student_model.parameters():
        param.requires_grad = False
    
    # Extract train embeddings
    print(f"Extracting training embeddings...")
    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=False,
        batch_size=eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset["train"].collate
    )
    train_embeddings, train_labels = extract_embeddings_efficient(
        student_model, train_dataloader, dataset["train"], device
    )
    print(f"  Train embeddings shape: {train_embeddings.shape}")
    
    # Determine evaluation set: prefer test, fallback to dev
    eval_set_name = "test" if "test" in dataset else "dev"
    eval_dataset = dataset[eval_set_name]
    
    print(f"Extracting {eval_set_name} embeddings...")
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_dataset.collate
    )
    eval_embeddings, eval_labels = extract_embeddings_efficient(
        student_model, eval_dataloader, eval_dataset, device
    )
    print(f"  {eval_set_name.upper()} embeddings shape: {eval_embeddings.shape}")
    
    # ========== STEP 2: For each dimension, train and evaluate ==========
    print("\n" + "="*60)
    print("Step 2: Training linear classifiers for each MRL dimension...")
    print("="*60)
    
    for dim_idx, dim in enumerate(matryoshka_dims):
        print(f"\n[{dim_idx+1}/{len(matryoshka_dims)}] Dimension: {dim}")
        print("="*60)
        
        # Create fresh linear head for this dimension
        clf_head = nn.Linear(dim, num_labels).to(device)
        nn.init.xavier_uniform_(clf_head.weight)
        nn.init.zeros_(clf_head.bias)
        optimizer = AdamW(clf_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Train linear head efficiently
        print(f"  Training linear head for {clf_epochs} epochs...")
        train_clf_head_batch_efficient(
            clf_head, train_embeddings, train_labels, dim, num_labels,
            optimizer, clf_epochs, args.batch_size, device
        )
        
        # Evaluation on eval set (test or dev)
        print(f"  Evaluating on {eval_set_name} set...")
        clf_head.eval()
        
        accuracy, precision, recall, avg_loss = evaluate_clf_batch_efficient(
            clf_head, eval_embeddings, eval_labels, dim, num_labels, device, eval_batch_size
        )
        
        print(f"  {eval_set_name.upper()} Results - Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | Loss: {avg_loss:.4f}")
        
        # Store results with eval set identifier
        all_dim_results[dim] = {
            f"{eval_set_name}_accuracy": round(float(accuracy), 6),
            f"{eval_set_name}_precision": round(float(precision), 6),
            f"{eval_set_name}_recall": round(float(recall), 6),
            f"{eval_set_name}_loss": round(float(avg_loss), 6),
        }
        
        # Clear GPU memory
        del clf_head, optimizer
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("Classification fine-tuning complete!")
    print("="*60)
    return all_dim_results


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
    
    # ========== STS Evaluation on Multiple Datasets ==========
    if args.do_train and hasattr(args, 'do_eval') and args.do_eval:
        print("\n" + "="*80)
        print("[1/3] STS EVALUATION ON MULTIPLE DATASETS".center(80, "="))
        print("="*80)
        
        # Try to load multiple STS datasets
        sts_eval_datasets = prepare_sts_datasets_list(args, distiller)
        
        if not sts_eval_datasets:
            # Fall back to single dataset if list is not provided
            print("\nAttempting to load single STS dataset...")
            sts_eval_datasets = {}
            sts_eval_data = prepare_sts_dataset(args, distiller)
            if sts_eval_data:
                sts_eval_datasets["default"] = sts_eval_data
        
        # Evaluate on all datasets
        all_sts_results = {}
        for dataset_name, sts_eval_data in sts_eval_datasets.items():
            print(f"\n{'='*60}")
            print(f"Evaluating STS dataset: {dataset_name}")
            print(f"{'='*60}")
            
            if "dev" in sts_eval_data:
                print(f"\nEvaluating on {dataset_name} dev set with MRL dimensions")
                dev_results = evaluate_sts(
                    args, 
                    distiller.student_tokenizer, 
                    model_engine.module.student_model, 
                    sts_eval_data["dev"], 
                    f"{dataset_name}_dev", 
                    device
                )
                all_sts_results[f"{dataset_name}_dev"] = dev_results
            
            if "test" in sts_eval_data:
                print(f"\nEvaluating on {dataset_name} test set with MRL dimensions")
                test_results = evaluate_sts(
                    args, 
                    distiller.student_tokenizer, 
                    model_engine.module.student_model, 
                    sts_eval_data["test"], 
                    f"{dataset_name}_test", 
                    device
                )
                all_sts_results[f"{dataset_name}_test"] = test_results
        
    
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
    
    # ========== CLF Fine-tuning on Multiple Datasets (Per Dimension) ==========
    print("\n" + "="*80)
    print("[2/3] CLF FINE-TUNING ON MULTIPLE DATASETS (PER MRL DIMENSION)".center(80, "="))
    print("="*80)
    
    # Prepare multiple CLF datasets for fine-tuning and evaluation
    clf_finetune_datasets = prepare_clf_datasets_list(args, distiller)
    
    if not clf_finetune_datasets:
        # Fall back to single dataset if list is not provided
        print("\nAttempting to load single CLF dataset for fine-tuning...")
        clf_finetune_datasets = {}
        clf_data = prepare_clf_dataset(args, distiller)
        if clf_data:
            clf_finetune_datasets["default"] = clf_data
    
    # Fine-tune and evaluate on each CLF dataset separately
    all_clf_finetune_results = {}
    for dataset_name, clf_data in clf_finetune_datasets.items():
        print(f"\n{'='*70}")
        print(f"CLF Fine-tuning Dataset: {dataset_name}")
        print(f"{'='*70}")
        
        if not clf_data or "train" not in clf_data or "dev" not in clf_data:
            print(f"Warning: {dataset_name} missing train or dev data, skipping")
            continue
        
        # Fine-tune linear classifiers for each MRL dimension
        dim_results = finetune_clf_per_dimension(
            args, 
            distiller.student_tokenizer, 
            model_engine.module.student_model, 
            clf_data, 
            device
        )
        
        all_clf_finetune_results[dataset_name] = dim_results
        print(f"\nClassification fine-tuning complete for {dataset_name}!")
    
    # ========== Print Summary for All CLF Results ==========
    print("\n" + "="*80)
    print("CLASSIFICATION FINE-TUNING SUMMARY".center(80, "="))
    print("="*80)
    
    for dataset_name, dim_results in all_clf_finetune_results.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}".center(80, "="))
        print(f"{'='*80}")
        
        # Determine which eval set was used (test or dev)
        if dim_results:
            first_metrics = next(iter(dim_results.values()))
            has_test = any("test_" in key for key in first_metrics.keys())
            eval_set_name = "TEST" if has_test else "DEV"
        else:
            eval_set_name = "DEV"
        
        print(f"\n{f'Evaluation Results ({eval_set_name} Set)':^80}")
        print(f"{'Dimension':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'Loss':<15}")
        print("-" * 80)
        
        for dim in sorted(dim_results.keys()):
            metrics = dim_results[dim]
            acc_key = f"{eval_set_name.lower()}_accuracy"
            prec_key = f"{eval_set_name.lower()}_precision"
            rec_key = f"{eval_set_name.lower()}_recall"
            loss_key = f"{eval_set_name.lower()}_loss"
            
            print(f"{dim:<15} {metrics[acc_key]:<15.4f} {metrics[prec_key]:<15.4f} {metrics[rec_key]:<15.4f} {metrics[loss_key]:<15.4f}")
    
    print("\n" + "="*80)
    print(" FINE-TUNING COMPLETE ".center(80, "="))
    print("="*80)
    
if __name__ == "__main__":
    main()
