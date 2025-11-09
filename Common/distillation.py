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
        log_rank("Num of train data: {}".format(len(data["train"])))
    else:
        raise ValueError("do_train must be set to True")
        
    return data

def prepare_sts_dataset(args, distiller):
    """Prepare STS dev/test datasets for evaluation after training"""
    sts_data = {}
    
    # Check if STS data directory exists
    sts_data_dir = getattr(args, 'sts_data_dir', None)
    if not sts_data_dir:
        log_rank("Warning: sts_data_dir not specified, skipping STS evaluation")
        return sts_data
    
    if not os.path.exists(sts_data_dir):
        log_rank(f"Warning: STS data directory {sts_data_dir} does not exist")
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
            log_rank("Num of STS dev data: {}".format(len(sts_data["dev"])))
        
        test_path = os.path.join(sts_data_dir, "test.csv")
        if os.path.exists(test_path):
            sts_data["test"] = STSDataset(
                args,
                "test",
                distiller.student_tokenizer,
                distiller.teacher_tokenizer
            )
            log_rank("Num of STS test data: {}".format(len(sts_data["test"])))
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
        log_rank("Warning: nli_data_dir not specified, skipping NLI evaluation")
        return nli_data
    
    if not os.path.exists(nli_data_dir):
        log_rank(f"Warning: NLI data directory {nli_data_dir} does not exist")
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
            log_rank("Num of NLI dev data: {}".format(len(nli_data["dev"])))
        
        test_path = os.path.join(nli_data_dir, "test.csv")
        if os.path.exists(test_path):
            nli_data["test"] = NLIDataset(
                args,
                "test",
                distiller.student_tokenizer,
                distiller.teacher_tokenizer
            )
            log_rank("Num of NLI test data: {}".format(len(nli_data["test"])))
    finally:
        # Restore original data_dir
        args.data_dir = original_data_dir
    
    return nli_data

def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device):
    log_rank("Start Contrastive Learning Training")
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
        
        log_rank("Start iterations of epoch {}".format(epoch + 1))
        model.train()
        log_rank("Training mode: {}".format(model.student_model.training))

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
                log_rank("⚠️ Loss is NaN or Inf. Skipping this step.")
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
            
            log_rank(f"Epoch {epoch + 1} Summary:")
            log_rank(f"  Average Loss: {avg_loss:.4f}")
            log_rank(f"  Average Accuracy: {avg_accuracy:.4f}")
            log_rank(f"  Samples Processed: {total_samples_global}")
            log_rank(f"  Time: {epoch_time:.2f}s")
            log_rank(f"  Throughput: {total_samples_global / epoch_time:.2f} samples/s")

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
                log_rank("Saving tokenizer...")
                tokenizer.save_pretrained(save_dir_path)
                log_rank("Saving model...")
                model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
                log_rank("Saving config...")
                model.module.student_model.config.save_pretrained(save_dir_path)
            
            if hasattr(model.module, "projectors"):
                log_rank("Saving projector...")
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
                log_rank(f"Removed checkpoint: {removed_model['path']}")

            log_rank(f"Model has been saved to {save_dir_path}")
        
        dist.barrier()
            
    total_seconds = time.time() - start_time
    log_rank("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))
    
    # Save final model
    if dist.get_rank() == 0 and args.save_dir:
        final_save_dir = os.path.join(args.save_dir, "final_model")
        os.makedirs(final_save_dir, exist_ok=True)
        
        log_rank("Saving final model...")
        tokenizer.save_pretrained(final_save_dir)
        model.module.student_model.save_pretrained(final_save_dir, safe_serialization=False)
        model.module.student_model.config.save_pretrained(final_save_dir)
        
        if hasattr(model.module, "projectors"):
            torch.save(
                model.module.projectors.state_dict(), 
                os.path.join(final_save_dir, "projector.pt")
            )
        
        log_rank(f"Final model saved to {final_save_dir}")


@torch.no_grad()
def evaluate_sts(args, tokenizer, student_model, dataset, split, device):
    """
    Evaluate model on STS tasks with Pearson and Spearman correlations.
    Mimics STS/distillation.py evaluate function.
    """
    if dist.get_rank() != 0:
        return None, None, None, None        
    
    # Use regular DataLoader without DistributedSampler
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    eval_info = {
        "loss": 0.0,
        "sample_num": 0
    }

    all_preds = []
    all_targets = []
    total_loss = 0
    
    for input_batch, output_batch in tqdm(dataloader, desc="Processing batches"):
        dataset.move_to_device([input_batch, output_batch], device)
        targets = output_batch["labels"]
        
        outputs = student_model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            token_type_ids=input_batch.get("token_type_ids", None)
        )
        
        # Get predictions - could be scores from a regression head or embeddings
        if hasattr(outputs, 'scores'):
            predictions = outputs.scores 
        elif hasattr(outputs, 'last_hidden_state'):
            # Use mean pooling for embeddings if no scores attribute
            last_hidden = outputs.last_hidden_state
            attention_mask_expanded = input_batch["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            # Use embedding norm scaled to [0, 5] as predictions
            predictions = torch.norm(embeddings, dim=-1, keepdim=True) / torch.norm(embeddings).max() * 5.0
        else:
            raise ValueError("Cannot extract predictions from model output")
        
        # Ensure predictions and targets have compatible shapes
        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(-1)
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(-1)
            
        # Compute MSE loss
        loss = F.mse_loss(predictions, targets)
        
        all_preds.append(predictions)
        all_targets.append(targets)
        sample_num = targets.size(0)
        total_loss += loss.item() * sample_num

        eval_info["sample_num"] += sample_num
        
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Convert to float32 before converting to numpy (BFloat16 is not supported by numpy)
    all_preds = all_preds.to(torch.float32)
    all_targets = all_targets.to(torch.float32)

    # Convert to numpy for correlation metrics
    all_preds_np = all_preds.cpu().numpy().flatten()
    all_targets_np = all_targets.cpu().numpy().flatten()

    # Calculate Pearson and Spearman correlations
    from scipy.stats import pearsonr, spearmanr
    pearson_correlation, _ = pearsonr(all_preds_np, all_targets_np)
    spearman_correlation, _ = spearmanr(all_preds_np, all_targets_np)
    
    # Update evaluation info
    eval_info["loss"] = float(total_loss / eval_info["sample_num"])
    eval_info["pearson"] = round(float(pearson_correlation), 6)
    eval_info["spearman"] = round(float(spearman_correlation), 6)
    eval_info["mse"] = round(float(((all_preds_np - all_targets_np) ** 2).mean()), 6)

    if hasattr(args, 'local_rank') and args.local_rank == 0 or not hasattr(args, 'local_rank'):
        log_rank(f"Evaluated: {split} | {eval_info}")

    student_model.train()

    return eval_info["loss"], eval_info["pearson"], eval_info["spearman"]

@torch.no_grad()
def evaluate_nli(args, tokenizer, student_model, dataset, split, device):
    """
    Evaluate model on NLI tasks with accuracy, precision, and recall.
    Mimics SentencePair/distillation.py evaluate function.
    """
    if dist.get_rank() != 0:
        return None, None, None, None        
    
    # Use regular DataLoader without DistributedSampler
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    eval_info = {
        "loss": 0.0,
        "sample_num": 0,
        "correct_samples": 0
    }

    all_preds = []
    all_labels = []
    total_loss = 0
    
    for input_batch, output_batch in tqdm(dataloader, desc="Processing batches"):
        dataset.move_to_device([input_batch, output_batch], device)
        labels = output_batch["labels"]
        
        outputs = student_model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            token_type_ids=input_batch.get("token_type_ids", None)
        )
        
        # Get predictions - check if model has logits (classification head) or embeddings
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            # Try to get loss if available
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Compute cross-entropy loss if not available
                loss = F.cross_entropy(logits, labels)
        else:
            raise ValueError("Cannot extract logits from model output for NLI classification")
        
        correct = (preds == labels).sum().item()
        all_preds.append(preds)
        all_labels.append(labels)
        sample_num = labels.size(0)
        total_loss += loss.item() * sample_num

        eval_info["sample_num"] += sample_num
        eval_info["correct_samples"] += correct

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / eval_info["sample_num"] if eval_info["sample_num"] > 0 else 0.0

    eval_info["precision"] = round(float(precision), 6)
    eval_info["recall"] = round(float(recall), 6)
    eval_info["loss"] = round(float(avg_loss), 6)
    eval_info["accuracy"] = round(float(accuracy), 6)

    if hasattr(args, 'local_rank') and args.local_rank == 0 or not hasattr(args, 'local_rank'):
        log_rank(f"Evaluated: {split} | {eval_info}")

    student_model.train()

    return eval_info["loss"], eval_info["accuracy"], eval_info["precision"], eval_info["recall"]

@torch.no_grad()
def evaluate_clf(args, tokenizer, student_model, dataset, split, device):
    """
    Evaluate model on classification tasks with accuracy, precision, and recall.
    Mimics Classification/distillation.py evaluate function.
    """
    if dist.get_rank() != 0:
        return None, None, None, None        
    
    # Use regular DataLoader without DistributedSampler
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )

    student_model.eval()
    eval_info = {
        "loss": 0.0,
        "sample_num": 0,
        "correct_samples": 0
    }

    all_preds = []
    all_labels = []
    total_loss = 0
    
    for input_batch, output_batch in tqdm(dataloader, desc="Processing batches"):
        dataset.move_to_device([input_batch, output_batch], device)
        labels = output_batch["labels"]
        
        outputs = student_model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            token_type_ids=input_batch.get("token_type_ids", None)
        )
        
        # Get predictions - check if model has logits (classification head)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            # Compute cross-entropy loss manually since BertModel doesn't accept labels
            loss = F.cross_entropy(logits, labels)
        else:
            raise ValueError("Cannot extract logits from model output for classification")
        
        correct = (preds == labels).sum().item()
        all_preds.append(preds)
        all_labels.append(labels)
        sample_num = labels.size(0)
        total_loss += loss.item() * sample_num

        eval_info["sample_num"] += sample_num
        eval_info["correct_samples"] += correct

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / eval_info["sample_num"] if eval_info["sample_num"] > 0 else 0.0

    eval_info["precision"] = round(float(precision), 6)
    eval_info["recall"] = round(float(recall), 6)
    eval_info["loss"] = round(float(avg_loss), 6)
    eval_info["accuracy"] = round(float(accuracy), 6)

    if hasattr(args, 'local_rank') and args.local_rank == 0 or not hasattr(args, 'local_rank'):
        log_rank(f"Evaluated: {split} | {eval_info}")

    student_model.train()

    return eval_info["loss"], eval_info["accuracy"], eval_info["precision"], eval_info["recall"]


def prepare_clf_dataset(args, distiller):
    """Prepare CLF train/dev datasets for classifier fine-tuning"""
    clf_data = {}
    
    # Check if CLF data directory exists
    clf_data_dir = getattr(args, 'clf_data_dir', None)
    if not clf_data_dir:
        log_rank("Warning: clf_data_dir not specified, skipping CLF fine-tuning")
        return clf_data
    
    if not os.path.exists(clf_data_dir):
        log_rank(f"Warning: CLF data directory {clf_data_dir} does not exist")
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
            log_rank("Num of CLF train data: {}".format(len(clf_data["train"])))
        
        # Load dev data
        dev_path = os.path.join(clf_data_dir, "dev.csv")
        if os.path.exists(dev_path):
            clf_data["dev"] = CLFDataset(
                args,
                "dev",
                distiller.student_tokenizer,
                distiller.teacher_tokenizer
            )
            log_rank("Num of CLF dev data: {}".format(len(clf_data["dev"])))
    finally:
        # Restore original data_dir
        args.data_dir = original_data_dir
    
    return clf_data


def finetune_clf(args, tokenizer, model_engine, dataset, device):
    """Fine-tune classifier head on CLF dataset"""
    log_rank("\n" + "="*50)
    log_rank("Start Classification Head Fine-tuning (frozen base model)")
    log_rank("="*50)
    start_time = time.time()

    if "train" not in dataset or "dev" not in dataset:
        log_rank("Warning: train or dev dataset missing for CLF fine-tuning")
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
        
        log_rank(f"\nCLF Fine-tuning Epoch {epoch + 1}/{clf_epochs}")
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
                log_rank("⚠️ Loss is NaN or Inf. Skipping this step.")
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
            
            log_rank(f"CLF Epoch {epoch + 1} Summary:")
            log_rank(f"  Average Loss: {avg_loss:.4f}")
            log_rank(f"  Average Accuracy: {avg_accuracy:.4f}")
            log_rank(f"  Samples: {total_samples_global}")
            log_rank(f"  Time: {epoch_time:.2f}s")
        
        dist.barrier()
    
    total_seconds = time.time() - start_time
    log_rank("\nClassification Fine-tuning Done in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))
    
    # Evaluate on CLF dev set
    log_rank("\nEvaluating on CLF dev set...")
    model_engine.eval()
    dev_loss, dev_accuracy, dev_precision, dev_recall = evaluate_clf_dev(
        args, 
        model_engine.module,
        dataset["dev"],
        device
    )
    log_rank(f"CLF Dev Results - Loss: {dev_loss}, Acc: {dev_accuracy}, Prec: {dev_precision}, Rec: {dev_recall}")


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
    log_rank('DeepSpeed config: {}'.format(ds_config))

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * dp_world_size

    log_rank("Initializing a distiller for contrastive learning...")
    distiller = Distiller(args, device)
    dataset = prepare_dataset(args, distiller)
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size))
        assert args.total_iters is not None or args.num_epochs is not None
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.num_epochs
        if args.num_epochs is None:
            args.num_epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)

        log_rank("Total iterations: {}".format(args.total_iters))
        log_rank("Number of epochs: {}".format(args.num_epochs))
        log_rank("Iterations per epoch: {}".format(args.train_iters_per_epoch))
        
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
    
    # Evaluate on STS dev/test sets after training if do_eval is enabled
    if args.do_train and hasattr(args, 'do_eval') and args.do_eval:
        log_rank("Preparing STS evaluation datasets...")
        sts_eval_data = prepare_sts_dataset(args, distiller)
        
        if "dev" in sts_eval_data:
            log_rank("Evaluating on STS dev set...")
            dev_loss, dev_pearson, dev_spearman = evaluate_sts(
                args, 
                distiller.student_tokenizer, 
                model_engine.module.student_model, 
                sts_eval_data["dev"], 
                "dev", 
                device
            )
        
        if "test" in sts_eval_data:
            log_rank("Evaluating on STS test set...")
            test_loss, test_pearson, test_spearman = evaluate_sts(
                args, 
                distiller.student_tokenizer, 
                model_engine.module.student_model, 
                sts_eval_data["test"], 
                "test", 
                device
            )
    
    # # Evaluate on NLI dev/test sets after training if do_eval is enabled
    # if args.do_train and hasattr(args, 'do_eval') and args.do_eval:
    #     log_rank("Preparing NLI evaluation datasets...")
    #     nli_eval_data = prepare_nli_dataset(args, distiller)
        
    #     if "dev" in nli_eval_data:
    #         log_rank("Evaluating on NLI dev set...")
    #         dev_loss, dev_accuracy, dev_precision, dev_recall = evaluate_nli(
    #             args, 
    #             distiller.student_tokenizer, 
    #             model_engine.module.student_model, 
    #             nli_eval_data["dev"], 
    #             "dev", 
    #             device
    #         )
        
    #     if "test" in nli_eval_data:
    #         log_rank("Evaluating on NLI test set...")
    #         test_loss, test_accuracy, test_precision, test_recall = evaluate_nli(
    #             args, 
    #             distiller.student_tokenizer, 
    #             model_engine.module.student_model, 
    #             nli_eval_data["test"], 
    #             "test", 
    #             device
    #         )
    
    # Evaluate on CLF dev/test sets after training if do_eval is enabled
    log_rank("\n" + "="*60)
    log_rank("Checking for CLF fine-tuning...")
    log_rank("="*60)
    clf_data = prepare_clf_dataset(args, distiller)
    
    if clf_data and "train" in clf_data and "dev" in clf_data:
        # Create classification head
        log_rank("Creating classification head with frozen base model...")
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
        
        log_rank("Classification fine-tuning complete!")
    else:
        log_rank("Skipping CLF fine-tuning (clf_data_dir not specified or missing train/dev data)")
    
if __name__ == "__main__":
    main()