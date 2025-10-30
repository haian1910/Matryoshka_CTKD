import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List

class Matryoshka_InfoNCE_Loss(nn.Module):
    def __init__(self, temperature: float = 0.05, relative_importance: List[float] = None):
        super(Matryoshka_InfoNCE_Loss, self).__init__()
        self.temperature = temperature
        # relative importance shape: [G]
        self.relative_importance = relative_importance
    
    def forward(self, query_embeddings_list, positive_embeddings_list):
        losses = []
        correct_list = []
        
        for query_emb, pos_emb in zip(query_embeddings_list, positive_embeddings_list):
            # Normalize embeddings
            query_emb = F.normalize(query_emb, p=2, dim=-1)
            pos_emb = F.normalize(pos_emb, p=2, dim=-1)
            
            batch_size = query_emb.size(0)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(query_emb, pos_emb.t()) / self.temperature
            
            # Labels: diagonal elements are positive pairs
            labels = torch.arange(batch_size, device=query_emb.device)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(similarity_matrix, labels)
            losses.append(loss)
            
            # Compute accuracy
            predictions = similarity_matrix.argmax(dim=-1)
            correct = (predictions == labels).float().sum()
            correct_list.append(correct)
        
        # Stack losses: [G]
        losses = torch.stack(losses)
        
        # Set relative_importance to 1 if not specified
        rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance, device=losses.device)
        
        # Apply relative importance weights
        weighted_losses = rel_importance * losses
        total_loss = weighted_losses.sum()
        
        return total_loss, correct_list

class InfoNCELoss(nn.Module):
    def __init__(self, args) -> None:
        super(InfoNCELoss, self).__init__()
        self.temperature = args.contrastive_temperature if hasattr(args, 'contrastive_temperature') else 0.05
        self.use_distributed = dist.is_initialized()
        
        # MRL specific parameters
        self.nesting_list = getattr(args, 'mrl_nesting_list', [16, 32, 64, 128, 256, 512, 768])
        self.mrl_efficient = getattr(args, 'mrl_efficient', False)
        self.use_mrl = getattr(args, 'use_mrl', False)
        
        # Relative importance weights for different nesting dimensions
        self.relative_importance = getattr(args, 'mrl_relative_importance', None)
        if self.relative_importance is None:
            # Default: give more weight to larger dimensions
            if self.mrl_efficient or not self.use_mrl:
                self.relative_importance = [1.0]  # Only one dimension in efficient mode or no MRL
            else:
                # Progressive weighting: smaller dimensions get less weight
                num_dims = len(self.nesting_list)
                self.relative_importance = [(i + 1) / num_dims for i in range(num_dims)]
        
        # Initialize Matryoshka InfoNCE loss
        self.matryoshka_loss = Matryoshka_InfoNCE_Loss(
            temperature=self.temperature,
            relative_importance=self.relative_importance
        )
        
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
       
        self.distiller = distiller
        model = distiller.student_model
        
        # Matryoshka mode: get embeddings at multiple granularities
        query_embeddings_dict = distiller.get_matryoshka_embeddings(
            model,
            input_data['query_input_ids'],
            input_data['query_attention_mask']
        )
        
        positive_embeddings_dict = distiller.get_matryoshka_embeddings(
            model,
            input_data['positive_input_ids'],
            input_data['positive_attention_mask']
        )
        
        # Compute Matryoshka InfoNCE loss
        loss, correct_list = self.compute_matryoshka_infonce_loss(
            query_embeddings_dict, 
            positive_embeddings_dict
        )
        
        # Use the largest dimension for logging correct predictions
        correct = correct_list[-1] if correct_list else torch.tensor(0.0)
        
        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": loss,
                "correct": correct
            }
        )
        return loss, logging_output

    def compute_matryoshka_infonce_loss(self, query_embeddings_dict, positive_embeddings_dict):
        
        # Regular mode: order by nesting dimensions
        query_list = []
        positive_list = []
        for nesting_size in self.nesting_list:
            key = f"emb_{nesting_size}"
            if key in query_embeddings_dict and key in positive_embeddings_dict:
                query_list.append(query_embeddings_dict[key])
                positive_list.append(positive_embeddings_dict[key])
        
        if not query_list:
            # Fallback: use all available embeddings
            query_list = list(query_embeddings_dict.values())
            positive_list = list(positive_embeddings_dict.values())
        
        # Handle distributed training: gather embeddings from all GPUs
        if self.use_distributed and dist.get_world_size() > 1:
            gathered_query_list = []
            gathered_positive_list = []
            
            for query_emb, pos_emb in zip(query_list, positive_list):
                gathered_query = self.gather_tensors(query_emb)
                gathered_pos = self.gather_tensors(pos_emb)
                gathered_query_list.append(gathered_query)
                gathered_positive_list.append(gathered_pos)
            
            # Calculate labels with offset for current rank
            rank = dist.get_rank()
            batch_size = query_list[0].size(0)
            labels_offset = rank * batch_size
            
            # Compute loss using local queries against all gathered positives
            losses = []
            correct_list = []
            
            for local_query, all_positive in zip(query_list, gathered_positive_list):
                # Normalize embeddings
                local_query = F.normalize(local_query, p=2, dim=-1)
                all_positive = F.normalize(all_positive, p=2, dim=-1)
                
                # Compute similarity matrix
                similarity_matrix = torch.matmul(local_query, all_positive.t()) / self.temperature
                
                # Labels with offset
                labels = torch.arange(batch_size, device=local_query.device) + labels_offset
                
                # Compute loss
                loss = F.cross_entropy(similarity_matrix, labels)
                losses.append(loss)
                
                # Compute accuracy
                predictions = similarity_matrix.argmax(dim=-1)
                correct = (predictions == labels).float().sum()
                correct_list.append(correct)
            
            # Apply relative importance weights
            losses = torch.stack(losses)
            rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance, device=losses.device)
            weighted_losses = rel_importance * losses
            total_loss = weighted_losses.sum()
            
        else:
            # Single GPU: use Matryoshka loss directly
            total_loss, correct_list = self.matryoshka_loss(query_list, positive_list)
        
        return total_loss, correct_list

    def compute_infonce_loss(self, query_embeddings, positive_embeddings):
        """
        Compute InfoNCE loss with in-batch negatives (regular mode).
        For each query, the positive is the corresponding positive sample,
        and negatives are all other positives in the batch.
        """
        batch_size = query_embeddings.size(0)
        
        # Gather embeddings from all GPUs if using distributed training
        if self.use_distributed and dist.get_world_size() > 1:
            # Gather all query and positive embeddings across GPUs
            all_query_embeddings = self.gather_tensors(query_embeddings)
            all_positive_embeddings = self.gather_tensors(positive_embeddings)
            
            # Calculate the offset for current rank's targets
            rank = dist.get_rank()
            local_batch_size = query_embeddings.size(0)
            labels_offset = rank * local_batch_size
        else:
            all_query_embeddings = query_embeddings
            all_positive_embeddings = positive_embeddings
            labels_offset = 0
        
        # Compute similarity matrix: (local_batch_size, global_batch_size)
        # Each query against all positives (from all GPUs)
        similarity_matrix = torch.matmul(
            query_embeddings, 
            all_positive_embeddings.t()
        ) / self.temperature
        
        # Labels: for each query, the correct positive is at the same index
        labels = torch.arange(batch_size, device=query_embeddings.device) + labels_offset
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        # Compute accuracy (how many queries correctly identify their positive)
        predictions = similarity_matrix.argmax(dim=-1)
        correct = (predictions == labels).float().sum()
        
        return loss, correct

    def gather_tensors(self, tensor):
        """
        Gather tensors from all GPUs for distributed training.
        """
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)

    def record_logging_output(self, logging_output, batch_denom, content):
        """
        Record metrics like loss and accuracy for logging, handling distributed training.
        content = {
                "loss": loss,
                "correct": correct
            }
        """
        
        for k, v in content.items():
            if k == "correct":
                # Sum the correct counts across processes
                if isinstance(v, torch.Tensor):
                    record_v = v.clone()
                    dist.all_reduce(record_v, dist.ReduceOp.SUM)
                    record_v = record_v.item()
                else:
                    record_v = v
            else:
                # Normalize loss by batch_denom and average across processes
                record_v = v / batch_denom
                dist.all_reduce(record_v, dist.ReduceOp.SUM)
                record_v = record_v.item() / dist.get_world_size()
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
        return logging_output
