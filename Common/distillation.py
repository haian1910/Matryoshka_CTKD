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

torch.set_num_threads(4)

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
    
if __name__ == "__main__":
    main()