import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class InfoNCELoss(nn.Module):
    def __init__(self, args) -> None:
        super(InfoNCELoss, self).__init__()
        self.temperature = args.temperature if hasattr(args, 'temperature') else 0.05
        self.use_distributed = dist.is_initialized()
        
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute InfoNCE contrastive loss for query-positive pairs.
        - Uses distiller's get_embeddings method for proper pooling
        - batch_denom is typically the batch size
        """
        self.distiller = distiller
        model = distiller.student_model
        
        # Get embeddings for queries using distiller's pooling method
        query_embeddings = distiller.get_embeddings(
            model,
            input_data['query_input_ids'],
            input_data['query_attention_mask']
        )
        
        # Get embeddings for positives
        positive_embeddings = distiller.get_embeddings(
            model,
            input_data['positive_input_ids'],
            input_data['positive_attention_mask']
        )
        
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
        
        # Compute InfoNCE loss
        loss, correct = self.compute_infonce_loss(query_embeddings, positive_embeddings)
        
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

    def compute_infonce_loss(self, query_embeddings, positive_embeddings):
        """
        Compute InfoNCE loss with in-batch negatives.
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
                record_v = v.clone()
                dist.all_reduce(record_v, dist.ReduceOp.SUM)
                record_v = record_v.item()
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