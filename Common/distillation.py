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

def parse_num_labels(num_labels_arg):
    """
    Parse num_labels argument which can be:
    - A single integer (e.g., '2' or 2)
    - A comma-separated list (e.g., '2,3,4')
    
    Returns a dict mapping index to num_labels, or a single int if only one value.
    """
    if isinstance(num_labels_arg, int):
        return num_labels_arg
    
    if isinstance(num_labels_arg, str):
        if ',' in num_labels_arg:
            # Multiple values for multiple datasets
            values = [int(x.strip()) for x in num_labels_arg.split(',')]
            return values
        else:
            # Single value
            return int(num_labels_arg)
    
    return num_labels_arg

def get_num_labels_for_dataset(num_labels_arg, dataset_index=0, num_datasets=1):
    """
    Get the num_labels value for a specific dataset.
    
    Args:
        num_labels_arg: Can be int or list of ints
        dataset_index: Index of the current dataset
        num_datasets: Total number of datasets
    
    Returns:
        int: Number of labels for this dataset
    """
    parsed = parse_num_labels(num_labels_arg)
    
    if isinstance(parsed, list):
        if dataset_index < len(parsed):
            return parsed[dataset_index]
        else:
            # If not enough values provided, use the last one
            return parsed[-1]
    else:
        # Single value, use for all datasets
        return parsed

def truncate_embeddings(embeddings, dim):
    """Truncate embeddings to specified dimension for MRL evaluation."""
    if dim >= embeddings.size(-1):
        return embeddings
    return embeddings[:, :dim]

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

def prepare_sts_datasets_list(args, distiller):
    """Prepare STS datasets from a list of directories"""
    sts_datasets = {}
    sts_data_dirs = getattr(args, 'sts_data_dirs', [])
    if not sts_data_dirs:
        return sts_datasets
    
    if isinstance(sts_data_dirs, str):
        sts_data_dirs = [sts_data_dirs]
    
    original_data_dir = args.data_dir
    for sts_data_dir in sts_data_dirs:
        if not os.path.exists(sts_data_dir):
            continue
        
        dataset_name = os.path.basename(sts_data_dir.rstrip('/'))
        args.data_dir = sts_data_dir
        
        try:
            dataset_dict = {}
            dev_path = os.path.join(sts_data_dir, "dev.csv")
            if os.path.exists(dev_path):
                dataset_dict["dev"] = STSDataset(args, "dev", distiller.student_tokenizer, distiller.teacher_tokenizer)
                print(f"  Loaded STS '{dataset_name}' dev: {len(dataset_dict['dev'])} samples")
            
            test_path = os.path.join(sts_data_dir, "test.csv")
            if os.path.exists(test_path):
                dataset_dict["test"] = STSDataset(args, "test", distiller.student_tokenizer, distiller.teacher_tokenizer)
                print(f"  Loaded STS '{dataset_name}' test: {len(dataset_dict['test'])} samples")
            
            if dataset_dict:
                sts_datasets[dataset_name] = dataset_dict
        except Exception as e:
            print(f"Error loading STS dataset from {sts_data_dir}: {e}")
            continue
    
    args.data_dir = original_data_dir
    return sts_datasets

@torch.no_grad()
def evaluate_sts(args, tokenizer, student_model, dataset, split, device):
    """Evaluate model on STS tasks"""
    if dist.get_rank() != 0:
        return None
    
    matryoshka_dims = getattr(args, 'matryoshka_dims', [16, 32, 64, 128, 256, 512, 768])
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=dataset.collate)
    student_model.eval()
    
    dim_results = {dim: {"preds": [], "targets": []} for dim in matryoshka_dims}
    
    for input_batch, output_batch in tqdm(dataloader, desc=f"Evaluating STS {split}"):
        dataset.move_to_device([input_batch, output_batch], device)
        targets = output_batch["labels"]
        
        outputs_sent1 = student_model(
            input_ids=input_batch["input_ids_sent1"],
            attention_mask=input_batch["attention_mask_sent1"],
            token_type_ids=input_batch.get("token_type_ids_sent1", None)
        )
        
        outputs_sent2 = student_model(
            input_ids=input_batch["input_ids_sent2"],
            attention_mask=input_batch["attention_mask_sent2"],
            token_type_ids=input_batch.get("token_type_ids_sent2", None)
        )
        
        # Mean pooling
        last_hidden_sent1 = outputs_sent1.last_hidden_state
        attention_mask_expanded_sent1 = input_batch["attention_mask_sent1"].unsqueeze(-1).expand(last_hidden_sent1.size()).float()
        sum_embeddings_sent1 = torch.sum(last_hidden_sent1 * attention_mask_expanded_sent1, 1)
        sum_mask_sent1 = torch.clamp(attention_mask_expanded_sent1.sum(1), min=1e-9)
        embeddings_sent1 = sum_embeddings_sent1 / sum_mask_sent1
        
        last_hidden_sent2 = outputs_sent2.last_hidden_state
        attention_mask_expanded_sent2 = input_batch["attention_mask_sent2"].unsqueeze(-1).expand(last_hidden_sent2.size()).float()
        sum_embeddings_sent2 = torch.sum(last_hidden_sent2 * attention_mask_expanded_sent2, 1)
        sum_mask_sent2 = torch.clamp(attention_mask_expanded_sent2.sum(1), min=1e-9)
        embeddings_sent2 = sum_embeddings_sent2 / sum_mask_sent2
        
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(-1)
        
        for dim in matryoshka_dims:
            truncated_emb_sent1 = truncate_embeddings(embeddings_sent1, dim)
            truncated_emb_sent2 = truncate_embeddings(embeddings_sent2, dim)
            cosine_sim = F.cosine_similarity(truncated_emb_sent1, truncated_emb_sent2, dim=1)
            predictions = (cosine_sim + 1) / 2 * 5.0
            predictions = predictions.unsqueeze(-1)
            dim_results[dim]["preds"].append(predictions)
            dim_results[dim]["targets"].append(targets)
    
    results = {}
    for dim in matryoshka_dims:
        all_preds = torch.cat(dim_results[dim]["preds"], dim=0).to(torch.float32).cpu().numpy().flatten()
        all_targets = torch.cat(dim_results[dim]["targets"], dim=0).to(torch.float32).cpu().numpy().flatten()
        
        pearson_corr, _ = pearsonr(all_preds, all_targets)
        spearman_corr, _ = spearmanr(all_preds, all_targets)
        mse = ((all_preds - all_targets) ** 2).mean()
        
        results[dim] = {
            "pearson": round(float(pearson_corr), 6),
            "spearman": round(float(spearman_corr), 6),
            "mse": round(float(mse), 6)
        }
        print(f"STS {split} | Dim {dim:4d} | Pearson: {results[dim]['pearson']:.4f} | Spearman: {results[dim]['spearman']:.4f} | MSE: {results[dim]['mse']:.4f}")
    
    student_model.train()
    return results

def prepare_clf_datasets_list(args, distiller):
    """Prepare CLF datasets from a list of directories and map num_labels for each"""
    clf_datasets = {}
    clf_num_labels_map = {}  # Map dataset_name -> num_labels
    
    clf_data_dirs = getattr(args, 'clf_data_dirs', [])
    if not clf_data_dirs:
        return clf_datasets, clf_num_labels_map
    
    if isinstance(clf_data_dirs, str):
        clf_data_dirs = [clf_data_dirs]
    
    num_labels_arg = getattr(args, 'num_labels', '2')
    parsed_num_labels = parse_num_labels(num_labels_arg)
    
    original_data_dir = args.data_dir
    for dataset_idx, clf_data_dir in enumerate(clf_data_dirs):
        if not os.path.exists(clf_data_dir):
            continue
        
        dataset_name = os.path.basename(clf_data_dir.rstrip('/'))
        args.data_dir = clf_data_dir
        
        # Get num_labels for this dataset
        dataset_num_labels = get_num_labels_for_dataset(parsed_num_labels, dataset_idx, len(clf_data_dirs))
        clf_num_labels_map[dataset_name] = dataset_num_labels
        
        print(f"Dataset '{dataset_name}' will use num_labels={dataset_num_labels}")
        
        try:
            dataset_dict = {}
            train_path = os.path.join(clf_data_dir, "train.csv")
            if os.path.exists(train_path):
                dataset_dict["train"] = CLFDataset(args, "train", distiller.student_tokenizer, distiller.teacher_tokenizer)
                print(f"  Loaded CLF '{dataset_name}' train: {len(dataset_dict['train'])} samples")
            
            dev_path = os.path.join(clf_data_dir, "dev.csv")
            if os.path.exists(dev_path):
                dataset_dict["dev"] = CLFDataset(args, "dev", distiller.student_tokenizer, distiller.teacher_tokenizer)
                print(f"  Loaded CLF '{dataset_name}' dev: {len(dataset_dict['dev'])} samples")
            
            if dataset_dict:
                clf_datasets[dataset_name] = dataset_dict
        except Exception as e:
            print(f"Error loading CLF dataset from {clf_data_dir}: {e}")
            continue
    
    args.data_dir = original_data_dir
    return clf_datasets, clf_num_labels_map

@torch.no_grad()
def extract_embeddings_efficient(model, dataloader, dataset, device):
    """Efficiently extract embeddings with mean pooling"""
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
        
        last_hidden = outputs.last_hidden_state
        attention_mask_expanded = input_batch["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        embeddings_list.append(embeddings.cpu())
        labels_list.append(labels.cpu())
    
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return embeddings, labels

@torch.no_grad()
def evaluate_clf_batch_efficient(model, embeddings_full, labels, dim, num_labels, device, eval_batch_size=256):
    """Efficiently evaluate model in batches"""
    all_preds = []
    total_loss = 0
    
    embeddings_truncated = truncate_embeddings(embeddings_full, dim)
    labels = labels.to(device)
    
    for i in range(0, len(embeddings_truncated), eval_batch_size):
        batch_emb = embeddings_truncated[i:i+eval_batch_size].to(device)
        batch_lbl = labels[i:i+eval_batch_size]
        
        logits = model(batch_emb)
        loss = F.cross_entropy(logits, batch_lbl)
        
        all_preds.append(logits.argmax(dim=-1).cpu())
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
    """Train linear classifier efficiently in batches"""
    train_emb = truncate_embeddings(train_emb_full, dim)
    train_lbl = train_labels.to(device)
    
    for epoch in range(clf_epochs):
        clf_head.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        indices = torch.randperm(len(train_emb))
        
        for i in range(0, len(train_emb), batch_size):
            batch_indices = indices[i:i+batch_size]
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
            
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f"    Epoch {epoch+1}/{clf_epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

def finetune_clf_per_dimension(args, tokenizer, student_model, dataset, device, num_labels=None):
    """
    Per-dimension classification fine-tuning for large datasets.
    
    For each dataset:
      1. Extract all embeddings from frozen model (once)
      2. For each MRL dimension:
         - Truncate embeddings
         - Train linear head
         - Evaluate on test/dev set (prefer test, fallback to dev)
    
    Args:
        args: Arguments object
        tokenizer: Tokenizer
        student_model: Model to fine-tune
        dataset: Dataset dictionary with "train" and "dev" keys
        device: Device to use
        num_labels: Number of labels for this classification task. If None, will be read from args.num_labels
    """
    print("\n" + "="*50)
    print("Start Per-Dimension Classification Fine-tuning")
    print("="*50)
    
    if "train" not in dataset or "dev" not in dataset:
        print("Warning: train or dev dataset missing")
        return {}
    
    matryoshka_dims = getattr(args, 'matryoshka_dims', [16, 32, 64, 128, 256, 512, 768])
    
    # Use provided num_labels or fallback to args.num_labels
    if num_labels is None:
        num_labels_arg = getattr(args, 'num_labels', '2')
        num_labels = get_num_labels_for_dataset(num_labels_arg, 0, 1)
    clf_epochs = getattr(args, 'clf_epochs', 100)
    eval_batch_size = getattr(args, 'eval_batch_size', 4)
    
    all_dim_results = {}
    
    # STEP 1: Extract embeddings from frozen model
    print("\n" + "="*60)
    print("Step 1: Extracting embeddings from frozen base model...")
    print("="*60)
    
    student_model.eval()
    for param in student_model.parameters():
        param.requires_grad = False
    
    print(f"Extracting training embeddings...")
    train_dataloader = DataLoader(dataset["train"], shuffle=False, batch_size=eval_batch_size, 
                                  num_workers=args.num_workers, collate_fn=dataset["train"].collate)
    train_embeddings, train_labels = extract_embeddings_efficient(student_model, train_dataloader, dataset["train"], device)
    print(f"  Train embeddings shape: {train_embeddings.shape}")
    
    eval_set_name = "test" if "test" in dataset else "dev"
    eval_dataset = dataset[eval_set_name]
    
    print(f"Extracting {eval_set_name} embeddings...")
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=eval_batch_size, 
                                 num_workers=args.num_workers, collate_fn=eval_dataset.collate)
    eval_embeddings, eval_labels = extract_embeddings_efficient(student_model, eval_dataloader, eval_dataset, device)
    print(f"  {eval_set_name.upper()} embeddings shape: {eval_embeddings.shape}")
    
    # STEP 2: For each dimension, train and evaluate
    print("\n" + "="*60)
    print("Step 2: Training linear classifiers for each MRL dimension...")
    print("="*60)
    
    for dim_idx, dim in enumerate(matryoshka_dims):
        print(f"\n[{dim_idx+1}/{len(matryoshka_dims)}] Dimension: {dim}")
        print("="*60)
        
        clf_head = nn.Linear(dim, num_labels).to(device)
        nn.init.xavier_uniform_(clf_head.weight)
        nn.init.zeros_(clf_head.bias)
        optimizer = AdamW(clf_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        print(f"  Training linear head for {clf_epochs} epochs...")
        train_clf_head_batch_efficient(clf_head, train_embeddings, train_labels, dim, num_labels,
                                       optimizer, clf_epochs, args.batch_size, device)
        
        print(f"  Evaluating on {eval_set_name} set...")
        clf_head.eval()
        accuracy, precision, recall, avg_loss = evaluate_clf_batch_efficient(
            clf_head, eval_embeddings, eval_labels, dim, num_labels, device, eval_batch_size)
        
        print(f"  {eval_set_name.upper()} Results - Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | Loss: {avg_loss:.4f}")
        
        all_dim_results[dim] = {
            f"{eval_set_name}_accuracy": round(float(accuracy), 6),
            f"{eval_set_name}_precision": round(float(precision), 6),
            f"{eval_set_name}_recall": round(float(recall), 6),
            f"{eval_set_name}_loss": round(float(avg_loss), 6),
        }
        
        del clf_head, optimizer
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("Classification fine-tuning complete!")
    print("="*60)
    return all_dim_results

# def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device):
#     """Main contrastive learning training loop (kept for backwards compatibility)"""
#     print("Start Contrastive Learning Training")
#     start_time = time.time()

#     dp_world_size = dist.get_world_size()
#     dp_rank = dist.get_rank()
#     criterion = build_criterion(args)

#     sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
#     train_loader = DataLoader(dataset['train'], sampler=sampler, batch_size=args.batch_size, 
#                               num_workers=args.num_workers, collate_fn=dataset["train"].collate)
    
#     model_list = []
    
#     for epoch in range(args.num_epochs):
#         sampler.set_epoch(epoch)
#         print(f"Epoch {epoch + 1}/{args.num_epochs}")
#         model.train()

#         epoch_start_time = time.time()
#         data_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}") if dist.get_rank() == 0 else train_loader

#         for batch in data_iter:
#             input_batch = batch
#             dataset["train"].move_to_device(input_batch, device)

#             loss, _ = model(criterion, {"input_batch": input_batch, "output_batch": None}, {}, loss_denom=1)
            
#             if torch.isnan(loss) or torch.isinf(loss):
#                 continue

#             model.backward(loss)
#             model.step()

#         dist.barrier()
    
#     total_seconds = time.time() - start_time
#     print("Done training in {:0>2}:{:0>2}:{:0>2}".format(
#         int(total_seconds // 3600), int(total_seconds % 3600 // 60), int(total_seconds % 60)))

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
def main():
    torch.backends.cudnn.enabled = False
    args = get_args()
    initialize(args)
    dp_world_size = dist.get_world_size()

    if dist.get_rank() == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    device = torch.cuda.current_device()

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_rank(f"\n\n{'='*30} Contrastive Learning EXP at {cur_time} {'='*30}")
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * dp_world_size

    print("Initializing distiller...")
    distiller = Distiller(args, device)
    dataset = prepare_dataset(args, distiller)
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size))
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.num_epochs
        if args.num_epochs is None:
            args.num_epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)

        print(f"Total iterations: {args.total_iters}, Epochs: {args.num_epochs}")
    
    optimizer_grouped_parameters = get_optimizer(args, distiller.student_model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer_grouped_parameters)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=distiller, optimizer=optimizer_grouped_parameters, lr_scheduler=lr_scheduler, mpu=None, config_params=ds_config)
    
    if args.do_train:
        finetune(args, distiller.student_tokenizer, model_engine, optimizer, lr_scheduler, dataset, device)
    
    # STS Evaluation
    if args.do_train and hasattr(args, 'do_eval') and args.do_eval:
        print("\n" + "="*80)
        print("[1/3] STS EVALUATION ON MULTIPLE DATASETS".center(80, "="))
        print("="*80)
        
        sts_eval_datasets = prepare_sts_datasets_list(args, distiller)
        all_sts_results = {}
        
        for dataset_name, sts_eval_data in sts_eval_datasets.items():
            print(f"\nEvaluating STS dataset: {dataset_name}")
            if "dev" in sts_eval_data:
                dev_results = evaluate_sts(args, distiller.student_tokenizer, 
                                          model_engine.module.student_model, sts_eval_data["dev"], f"{dataset_name}_dev", device)
                all_sts_results[f"{dataset_name}_dev"] = dev_results
                print(dev_results)
            
            if "test" in sts_eval_data:
                test_results = evaluate_sts(args, distiller.student_tokenizer, 
                                           model_engine.module.student_model, sts_eval_data["test"], f"{dataset_name}_test", device)
                all_sts_results[f"{dataset_name}_test"] = test_results
                print(test_results)
    
    # CLF Fine-tuning
    print("\n" + "="*80)
    print("[2/3] CLF FINE-TUNING ON MULTIPLE DATASETS (PER MRL DIMENSION)".center(80, "="))
    print("="*80)
    
    clf_finetune_datasets, clf_num_labels_map = prepare_clf_datasets_list(args, distiller)
    all_clf_finetune_results = {}
    
    for dataset_name, clf_data in clf_finetune_datasets.items():
        print(f"\n{'='*70}")
        print(f"CLF Fine-tuning Dataset: {dataset_name}")
        print(f"{'='*70}")
        
        if not clf_data or "train" not in clf_data or "dev" not in clf_data:
            print(f"Warning: {dataset_name} missing train or dev data")
            continue
        
        dataset_num_labels = clf_num_labels_map.get(dataset_name, 2)
        dim_results = finetune_clf_per_dimension(args, distiller.student_tokenizer, 
                                               model_engine.module.student_model, clf_data, device, dataset_num_labels)
        all_clf_finetune_results[dataset_name] = dim_results
        print(f"\nClassification fine-tuning complete for {dataset_name}!")
    
    # Summary
    print("\n" + "="*80)
    print("CLASSIFICATION FINE-TUNING SUMMARY".center(80, "="))
    print("="*80)
    
    for dataset_name, dim_results in all_clf_finetune_results.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}".center(80, "="))
        print(f"{'='*80}")
        
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
