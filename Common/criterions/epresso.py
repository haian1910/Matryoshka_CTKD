import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import random
from typing import List, Dict, Any


class MatryoshkaContrastiveLoss(nn.Module):
    """
    Matryoshka loss adapted for contrastive learning with InfoNCE.
    This implements the core idea of training at multiple embedding dimensions.
    """
    def __init__(
        self,
        matryoshka_dims: List[int],
        matryoshka_weights: List[float] = None,
        n_dims_per_step: int = -1,
        temperature: float = 0.05,
    ):
        super().__init__()
        if matryoshka_weights is None:
            num_dims = len(matryoshka_dims)
            matryoshka_weights = [(i + 1) / num_dims for i in range(num_dims)]
        
        # Sort dimensions and weights in descending order
        dims_weights = zip(matryoshka_dims, matryoshka_weights)
        self.matryoshka_dims, self.matryoshka_weights = zip(*sorted(dims_weights, key=lambda x: x[0], reverse=True))
        self.n_dims_per_step = n_dims_per_step
        self.temperature = temperature
        
    def forward(
        self, 
        query_embeddings_dict: Dict[str, torch.Tensor], 
        positive_embeddings_dict: Dict[str, torch.Tensor],
        use_distributed: bool = False
    ) -> tuple:
        """
        Compute weighted InfoNCE loss across different matryoshka dimensions.
        
        Args:
            query_embeddings_dict: Dictionary with keys like 'emb_768', 'emb_512', etc.
            positive_embeddings_dict: Dictionary with keys like 'emb_768', 'emb_512', etc.
            use_distributed: Whether to gather embeddings across GPUs
        
        Returns:
            total_loss: Weighted sum of losses
            correct_dict: Dictionary of correct predictions for each dimension
        """
        # Determine which dimensions to use in this step
        dim_indices = range(len(self.matryoshka_dims))
        if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
            dim_indices = random.sample(list(dim_indices), self.n_dims_per_step)
        
        total_loss = 0.0
        correct_dict = {}
        
        for idx in dim_indices:
            dim = self.matryoshka_dims[idx]
            weight = self.matryoshka_weights[idx]
            
            emb_key = f"emb_{dim}"
            if emb_key in query_embeddings_dict and emb_key in positive_embeddings_dict:
                query_emb = query_embeddings_dict[emb_key]
                pos_emb = positive_embeddings_dict[emb_key]
                
                # Normalize embeddings
                query_emb = F.normalize(query_emb, p=2, dim=-1)
                pos_emb = F.normalize(pos_emb, p=2, dim=-1)
                
                batch_size = query_emb.size(0)
                
                # Handle distributed training
                if use_distributed and dist.is_initialized() and dist.get_world_size() > 1:
                    # Gather embeddings from all GPUs
                    all_pos_emb = self.gather_tensors(pos_emb)
                    
                    # Calculate labels with offset
                    rank = dist.get_rank()
                    labels_offset = rank * batch_size
                    labels = torch.arange(batch_size, device=query_emb.device) + labels_offset
                else:
                    all_pos_emb = pos_emb
                    labels = torch.arange(batch_size, device=query_emb.device)
                
                # Compute similarity matrix
                similarity_matrix = torch.matmul(query_emb, all_pos_emb.t()) / self.temperature
                
                # Compute InfoNCE loss
                loss = F.cross_entropy(similarity_matrix, labels)
                total_loss += weight * loss
                
                # Compute accuracy
                predictions = similarity_matrix.argmax(dim=-1)
                correct = (predictions == labels).float().sum()
                correct_dict[emb_key] = correct
        
        # Normalize by number of dimensions used
        if len(list(dim_indices)) > 0:
            total_loss = total_loss / len(list(dim_indices))
        
        return total_loss, correct_dict
    
    def gather_tensors(self, tensor):
        """Gather tensors from all GPUs for distributed training."""
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)


class AdaptiveLayerContrastiveLoss(nn.Module):
    """
    Adaptive layer loss adapted for contrastive learning.
    This trains intermediate layers to produce good embeddings.
    """
    def __init__(
        self,
        n_layers_per_step: int = 1,
        last_layer_weight: float = 1.0,
        prior_layers_weight: float = 1.0,
        kl_div_weight: float = 0.0,  # Typically 0 for contrastive learning
        kl_temperature: float = 0.3,
    ):
        super().__init__()
        self.n_layers_per_step = n_layers_per_step
        self.last_layer_weight = last_layer_weight
        self.prior_layers_weight = prior_layers_weight
        self.kl_div_weight = kl_div_weight
        self.kl_temperature = kl_temperature
    
    def forward(
        self, 
        distiller,
        query_input_ids,
        query_attention_mask,
        positive_input_ids,
        positive_attention_mask,
        base_loss_fn,
        matryoshka_dims: List[int]
    ):
        """
        Compute contrastive loss across different layers of the model.
        """
        model = distiller.student_model
        
        # Get hidden states for queries
        if hasattr(model, 'model'):  # For models like LLM2Vec
            query_outputs = model.model(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                output_hidden_states=True
            )
        elif hasattr(model, 'bert'):  # For BERT
            query_outputs = model.bert(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                output_hidden_states=True
            )
        else:
            query_outputs = model(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                output_hidden_states=True
            )
        
        # Get hidden states for positives
        if hasattr(model, 'model'):
            pos_outputs = model.model(
                input_ids=positive_input_ids,
                attention_mask=positive_attention_mask,
                output_hidden_states=True
            )
        elif hasattr(model, 'bert'):
            pos_outputs = model.bert(
                input_ids=positive_input_ids,
                attention_mask=positive_attention_mask,
                output_hidden_states=True
            )
        else:
            pos_outputs = model(
                input_ids=positive_input_ids,
                attention_mask=positive_attention_mask,
                output_hidden_states=True
            )
        
        query_hidden_states = query_outputs.hidden_states
        pos_hidden_states = pos_outputs.hidden_states
        num_layers = len(query_hidden_states) - 1  # Exclude embedding layer
        
        total_loss = 0.0
        
        # Process final layer with full MRL
        final_query_embeddings_dict = self.get_matryoshka_embeddings_from_hidden(
            query_hidden_states[-1], 
            query_attention_mask, 
            matryoshka_dims
            #distiller.args.pooling_method
        )
        final_pos_embeddings_dict = self.get_matryoshka_embeddings_from_hidden(
            pos_hidden_states[-1], 
            positive_attention_mask, 
            matryoshka_dims,
            #distiller.args.pooling_method
        )
        
        # Compute loss for final layer
        final_loss, final_correct_dict = base_loss_fn(
            final_query_embeddings_dict, 
            final_pos_embeddings_dict,
            use_distributed=True
        )
        total_loss += self.last_layer_weight * final_loss
        
        # Sample layers to train
        layer_indices = list(range(1, num_layers))  # Skip embedding layer and final layer
        if self.n_layers_per_step > 0 and self.n_layers_per_step < len(layer_indices):
            layer_indices = random.sample(layer_indices, self.n_layers_per_step)
        
        # Process sampled intermediate layers
        for layer_idx in layer_indices:
            layer_query_embeddings_dict = self.get_matryoshka_embeddings_from_hidden(
                query_hidden_states[layer_idx], 
                query_attention_mask, 
                matryoshka_dims
                #distiller.args.pooling_method
            )
            layer_pos_embeddings_dict = self.get_matryoshka_embeddings_from_hidden(
                pos_hidden_states[layer_idx], 
                positive_attention_mask, 
                matryoshka_dims
                #distiller.args.pooling_method
            )
            
            # Compute loss for this layer
            layer_loss, _ = base_loss_fn(
                layer_query_embeddings_dict, 
                layer_pos_embeddings_dict,
                use_distributed=True
            )
            
            # Weight by layer position (earlier layers get less weight)
            weight_factor = (layer_idx + 1) / num_layers
            total_loss += self.prior_layers_weight * weight_factor * layer_loss / len(layer_indices)
        
        return total_loss, final_query_embeddings_dict, final_pos_embeddings_dict, final_correct_dict
    
    def get_matryoshka_embeddings_from_hidden(
        self, 
        hidden_state, 
        attention_mask, 
        matryoshka_dims
    ):
        """Extract matryoshka embeddings from hidden states."""
        # # Apply pooling
        # if pooling_method == "cls":
        #     pooled = hidden_state[:, 0, :]
        # elif pooling_method == "mean":
        #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        #     sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        #     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        #     pooled = sum_embeddings / sum_mask
        # elif pooling_method == "last":
        #     sequence_lengths = attention_mask.sum(dim=1) - 1
        #     batch_size = hidden_state.shape[0]
        #     pooled = hidden_state[torch.arange(batch_size), sequence_lengths]
        # else:
        #     pooled = hidden_state[:, 0, :]
        
        # apply mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        # Create embeddings dict for different dimensions
        embeddings_dict = {}
        for dim in matryoshka_dims:
            embeddings_dict[f"emb_{dim}"] = pooled[:, :dim]
        
        return embeddings_dict


class EPRESSO(nn.Module):
    def __init__(self, args) -> None:
        super(EPRESSO, self).__init__()
        self.args = args
        
        # Matryoshka configuration
        self.matryoshka_dims = getattr(args, 'mrl_nesting_list', [16, 32, 64, 128, 256, 512, 768])
        self.matryoshka_weights = getattr(args, 'matryoshka_weights', None)
        self.n_dims_per_step = getattr(args, 'n_dims_per_step', -1)
        self.temperature = getattr(args, 'contrastive_temperature', 0.05)
        
        # Adaptive layer configuration
        self.use_adaptive_layers = getattr(args, 'use_adaptive_layers', False)
        self.n_layers_per_step = getattr(args, 'n_layers_per_step', 1)
        self.last_layer_weight = getattr(args, 'last_layer_weight', 1.0)
        self.prior_layers_weight = getattr(args, 'prior_layers_weight', 0.3)
        self.kl_div_weight = getattr(args, 'kl_div_weight', 0.0)
        self.kl_temperature = getattr(args, 'kl_temperature', 0.3)
        
        # Initialize loss components
        self.matryoshka_loss = MatryoshkaContrastiveLoss(
            matryoshka_dims=self.matryoshka_dims,
            matryoshka_weights=self.matryoshka_weights,
            n_dims_per_step=self.n_dims_per_step,
            temperature=self.temperature
        )
        
        if self.use_adaptive_layers:
            self.adaptive_layer_loss = AdaptiveLayerContrastiveLoss(
                n_layers_per_step=self.n_layers_per_step,
                last_layer_weight=self.last_layer_weight,
                prior_layers_weight=self.prior_layers_weight,
                kl_div_weight=self.kl_div_weight,
                kl_temperature=self.kl_temperature
            )
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute EPRESSO loss for contrastive learning.
        
        Args:
            distiller: The distiller module containing student and teacher models
            input_data: Dictionary with query and positive input_ids and attention_masks
            output_data: Not used for contrastive learning (can be None)
            logging_output: Dictionary for logging metrics
            batch_denom: Batch size for normalization
        """
        self.distiller = distiller
        
        if self.use_adaptive_layers:
            # Use adaptive layer loss (2D Matryoshka: dimensions + layers)
            loss, query_embeddings_dict, pos_embeddings_dict, correct_dict = self.adaptive_layer_loss(
                distiller=distiller,
                query_input_ids=input_data["query_input_ids"],
                query_attention_mask=input_data["query_attention_mask"],
                positive_input_ids=input_data["positive_input_ids"],
                positive_attention_mask=input_data["positive_attention_mask"],
                base_loss_fn=self.matryoshka_loss,
                matryoshka_dims=self.matryoshka_dims
            )
        else:
            # Standard forward pass with Matryoshka loss only (1D Matryoshka: dimensions)
            # Get embeddings at multiple dimensions
            query_embeddings_dict = distiller.get_matryoshka_embeddings(
                distiller.student_model,
                input_data["query_input_ids"],
                input_data["query_attention_mask"]
            )
            
            pos_embeddings_dict = distiller.get_matryoshka_embeddings(
                distiller.student_model,
                input_data["positive_input_ids"],
                input_data["positive_attention_mask"]
            )
            
            # Apply Matryoshka contrastive loss
            loss, correct_dict = self.matryoshka_loss(
                query_embeddings_dict, 
                pos_embeddings_dict,
                use_distributed=True
            )
        
        # Get the largest dimension for primary metrics
        max_dim = max(self.matryoshka_dims)
        max_dim_key = f"emb_{max_dim}"
        correct = correct_dict.get(max_dim_key, torch.tensor(0.0))
        
        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": loss,
                "correct": correct
            }
        )
        
        # Log individual dimension metrics if requested
        if getattr(self.args, 'log_dimension_losses', False):
            with torch.no_grad():
                for dim in self.matryoshka_dims:
                    dim_key = f"emb_{dim}"
                    if dim_key in correct_dict:
                        dim_correct = correct_dict[dim_key]
                        logging_output = self.record_logging_output(
                            logging_output,
                            batch_denom,
                            {f"correct_dim_{dim}": dim_correct}
                        )
        
        return loss, logging_output
    
    def record_logging_output(self, logging_output, batch_denom, content):
        """
        Record metrics like loss and accuracy for logging, handling distributed training.
        """
        for k, v in content.items():
            if k == "correct" or "correct_dim" in k:
                # Sum the correct counts across processes
                if isinstance(v, torch.Tensor):
                    record_v = v.clone()
                    if dist.is_initialized():
                        dist.all_reduce(record_v, dist.ReduceOp.SUM)
                    record_v = record_v.item()
                else:
                    record_v = v
            else:
                # Normalize loss by batch_denom and average across processes
                if isinstance(v, torch.Tensor):
                    record_v = v / batch_denom
                    if dist.is_initialized():
                        dist.all_reduce(record_v, dist.ReduceOp.SUM)
                        record_v = record_v.item() / dist.get_world_size()
                else:
                    record_v = v / batch_denom
                    
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
        return logging_output
