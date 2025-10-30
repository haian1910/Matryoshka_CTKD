import torch
import torch.nn as nn
from .matry_infoNCE import Matryoshka_InfoNCE_Loss
import math
import random
from typing import List, Optional
from torch.nn.utils.parametrizations import orthogonal

class OrthogonalProjection(nn.Module):
    """Orthogonal projection layer using PyTorch's built-in orthogonal parameterization.
    
    Maps student embedding to teacher dimension using guaranteed orthogonal matrices.
    Uses torch.nn.utils.parametrizations.orthogonal for true orthogonality.
    """
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Create linear layer
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
        # Apply orthogonal parameterization to ensure the weight matrix is always orthogonal
        # This automatically maintains orthogonality during training
        orthogonal(self.linear, 'weight')
        
    def forward(self, x):
        """Apply orthogonal projection."""
        # Ensure the linear layer is on the same device and dtype as input
        if (x.device != self.linear.weight.device or 
            x.dtype != self.linear.weight.dtype):
            self.linear = self.linear.to(device=x.device, dtype=x.dtype)
        
        return self.linear(x)


class CKALoss(nn.Module):
    """CKA Loss for measuring similarity between hidden representations"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, SH, TH): 
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(SH.device, torch.float64)
        
        # Center the representations
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
        
        # Compute CKA similarity
        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        
        return num / torch.sqrt(den1 * den2)

class MatryoshkaHiddenStateProcessor:
    """
    Processor for creating Matryoshka-style sub-matrices from hidden states (this will use later)
    Inspired by MatryoshkaLoss gradient handling approach
    """
    def __init__(self, matryoshka_dims: List[int], matryoshka_weights: Optional[List[float]] = None, 
                 n_dims_per_step: int = -1):
        self.matryoshka_dims = sorted(matryoshka_dims, reverse=True)  # Sort descending
        self.matryoshka_weights = matryoshka_weights or [1.0] * len(matryoshka_dims)
        self.n_dims_per_step = n_dims_per_step
        
        # Ensure weights match dimensions
        if len(self.matryoshka_weights) != len(self.matryoshka_dims):
            raise ValueError("matryoshka_weights must have same length as matryoshka_dims")
    
    def shrink_hidden_state(self, hidden_state: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Shrink hidden state to specified dimension and normalize
        Args:
            hidden_state: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            dim: target dimension
        """
        hidden_dim = hidden_state.shape[-1]
        if dim > hidden_dim:
            raise ValueError(f"Dimension {dim} cannot be greater than hidden dimension: {hidden_dim}")
        
        # Truncate to target dimension
        truncated = hidden_state[..., :dim]
        
        # L2 normalize along the last dimension
        normalized = torch.nn.functional.normalize(truncated, p=2, dim=-1)
        return normalized
    
    def process_hidden_states(self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, 
                            cka_loss_fn: CKALoss) -> torch.Tensor:
        """
        Process hidden states with Matryoshka-style dimensionality reduction
        Apply CKA loss for each sub-matrix dimension
        
        Args:
            student_hidden: Student hidden states [batch_size, seq_len, student_dim]
            teacher_hidden: Teacher hidden states [batch_size, seq_len, teacher_dim] 
            cka_loss_fn: CKA loss function
        
        Returns:
            Combined CKA loss across all dimensions
        """
        # Determine which dimensions to process
        dim_indices = list(range(len(self.matryoshka_dims)))
        if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
            dim_indices = random.sample(dim_indices, self.n_dims_per_step)
            dim_indices.sort()  # Keep order for consistency
        
        total_loss = 0.0
        
        for idx in dim_indices:
            dim = self.matryoshka_dims[idx]
            weight = self.matryoshka_weights[idx]
            
            # Check if dimension is valid for student hidden states
            if dim > student_hidden.shape[-1]:
                continue
                
            # Create sub-matrix from student hidden states
            compute_gradients = torch.is_grad_enabled()
            truncated_student = self.shrink_hidden_state(student_hidden, dim)
            
            # Handle gradients properly (inspired by MatryoshkaLoss)
            if compute_gradients:
                matryoshka_student = truncated_student.detach().requires_grad_()
            else:
                matryoshka_student = truncated_student
            
            # Compute CKA similarity
            cka_similarity = cka_loss_fn(matryoshka_student, teacher_hidden)
            
            # Loss is 1 - sqrt(CKA) as per your idea
            cka_loss = 1.0 - (cka_similarity)
            
            # Weight the loss
            weighted_loss = weight * cka_loss
            total_loss += weighted_loss
            
            # Propagate gradients back through truncation (inspired by MatryoshkaLoss)
            if compute_gradients and matryoshka_student.grad is not None:
                truncated_student.backward(weight * matryoshka_student.grad)
        
        return total_loss

class MATRY_CTKD(Matryoshka_InfoNCE_Loss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate  # Knowledge distillation rate
        self.nesting_list = getattr(args, 'mrl_nesting_list', [64, 128, 256, 512, 768])
        self.mrl_efficient = getattr(args, 'mrl_efficient', False)
        
        # Student layers to process for CKA loss
        self.student_layers_to_process = getattr(args, 'student_layers_to_process', [2, 7, 11])
        
        # Matryoshka dimensions for hidden state processing
        # These should be smaller than the student model's hidden dimension
        self.matryoshka_hidden_dims = getattr(args, 'matryoshka_hidden_dims', [64, 128, 256, 512, 768])
        self.matryoshka_hidden_weights = getattr(args, 'matryoshka_hidden_weights', [1.0 / (1 + math.log(idx + 1)) for idx in range(len(self.matryoshka_hidden_dims))])
        self.n_dims_per_step = getattr(args, 'n_dims_per_step', -1)
        
        # Initialize CKA loss function
        self.cka_loss_fn = CKALoss()
        
        # Initialize Matryoshka hidden state processor
        self.hidden_processor = MatryoshkaHiddenStateProcessor(
            matryoshka_dims=self.matryoshka_hidden_dims,
            matryoshka_weights=self.matryoshka_hidden_weights,
            n_dims_per_step=self.n_dims_per_step
        )
        
        # Relative importance weights for different nesting dimensions 
        self.relative_importance = getattr(args, 'mrl_relative_importance', None)
        if self.relative_importance is None:
            if self.mrl_efficient:
                self.relative_importance = [1.0]
            else:
                num_dims = len(self.nesting_list)
                self.relative_importance = [1.0 / (1 + math.log(i + 1)) for i in range(num_dims)]

        self.projectors = nn.ModuleDict()
        
        for dim in self.nesting_list:
            self.projectors[f'proj_{dim}'] = OrthogonalProjection(
                in_dim=dim, 
                out_dim=self.teacher_hidden_size
            )
    
    def to(self, device, dtype=None):
        """Override to method to ensure projectors are moved to the correct device and dtype"""
        result = super().to(device, dtype=dtype)
        # Ensure projectors are also moved
        for proj in self.projectors.values():
            proj.to(device, dtype=dtype)
        return result
    
    def cuda(self, device=None):
        """Override cuda method to ensure projectors are moved to CUDA"""
        result = super().cuda(device)
        # Ensure projectors are also moved
        for proj in self.projectors.values():
            proj.cuda(device)
        return result
        
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

        loss_task, correct_list = self.compute_matryoshka_infonce_loss(
            query_embeddings_dict, 
            positive_embeddings_dict
        )
        
        
        # Compute Matryoshka CTKD loss
        loss_kd = self.compute_matry_ctkd(
            query_embeddings_dict, 
            positive_embeddings_dict
        )
        
        # Use the largest dimension for logging correct predictions
        correct = correct_list[-1] if correct_list else torch.tensor(0.0)

        final_loss = (1.0 - self.kd_rate) * loss_task + self.kd_rate * loss_kd
        
        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": final_loss,
                "correct": correct
            }
        )
        return final_loss, logging_output

    
       
    def get_embedding_layer(self, model, model_type):
        """Extract embedding layer from different model architectures"""
        # Handle wrapped models (like BertWithMRLWrapper)
        if hasattr(model, 'bert'):
            base_model = model.bert
        elif hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model
            
        # Try different embedding layer access patterns
        if hasattr(base_model, "get_input_embeddings"):
            return base_model.get_input_embeddings()
        elif hasattr(base_model, "embeddings") and hasattr(base_model.embeddings, "word_embeddings"):
            return base_model.embeddings.word_embeddings
        elif hasattr(base_model, "embed_tokens"):
            return base_model.embed_tokens
        elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "wte"):
            return base_model.transformer.wte
        elif hasattr(base_model, "wte"):
            return base_model.wte
        else:
            raise NotImplementedError(f"Unsupported {model_type} model architecture for embedding extraction")


    def compute_matry_ctkd(self, query_embeddings, positive_embeddings):
        """
        Compute Matryoshka CTKD loss using orthogonal projections and InfoNCE.
        
        Following the formula from the paper:
        L_Matry-CTKD = (1/B) * sum_{b=1}^{B} sum_{k in K} sum_{i=1}^{S} w_i * l_NCE^{(k,i,b)}
        
        Where:
        - B is batch size
        - K is the set of student layers (in this case we use query/positive pairs)
        - S is the number of Matryoshka dimensions
        - w_i is the weight for dimension i
        - l_NCE^{(k,i,b)} is the InfoNCE loss for layer k, width i, sample b
        
        Args:
            query_embeddings: Dict mapping dimension -> query embeddings [batch_size, dim]
            positive_embeddings: Dict mapping dimension -> positive embeddings [batch_size, dim]
        
        Returns:
            Total Matryoshka CTKD loss
        """
        # Get teacher embeddings (mean pooled last layer)
        teacher_query = self.distiller.get_teacher_embeddings(
            self.distiller.teacher_model,
            self.distiller.teacher_input_ids_query,
            self.distiller.teacher_attention_mask_query
        )  # [batch_size, teacher_dim]
        
        teacher_positive = self.distiller.get_teacher_embeddings(
            self.distiller.teacher_model,
            self.distiller.teacher_input_ids_positive,
            self.distiller.teacher_attention_mask_positive
        )  # [batch_size, teacher_dim]
        
        # Normalize teacher embeddings
        teacher_query = torch.nn.functional.normalize(teacher_query, p=2, dim=-1)
        teacher_positive = torch.nn.functional.normalize(teacher_positive, p=2, dim=-1)
        
        batch_size = teacher_query.size(0)
        total_loss = 0.0
        
        # Determine which dimensions to process (for efficiency)
        if self.mrl_efficient:
            # Only use the largest dimension
            dim_indices = [len(self.nesting_list) - 1]
        else:
            dim_indices = list(range(len(self.nesting_list)))
        
        # Iterate over each Matryoshka dimension
        for idx in dim_indices:
            dim = self.nesting_list[idx]
            weight = self.relative_importance[idx]
            
            # Skip if dimension not available
            if dim not in query_embeddings or dim not in positive_embeddings:
                continue
            
            # Get student embeddings at this dimension
            student_query = query_embeddings[dim]  # [batch_size, dim]
            student_positive = positive_embeddings[dim]  # [batch_size, dim]
            
            # Truncate to ensure correct dimension (in case of padding)
            student_query = student_query[:, :dim]
            student_positive = student_positive[:, :dim]
            
            # Normalize student embeddings (important for InfoNCE)
            student_query = torch.nn.functional.normalize(student_query, p=2, dim=-1)
            student_positive = torch.nn.functional.normalize(student_positive, p=2, dim=-1)
            
            # Apply orthogonal projection: P_i * s_k^(i)(x) -> R^D
            projector = self.projectors[f'proj_{dim}']
            projected_query = projector(student_query)  # [batch_size, teacher_dim]
            projected_positive = projector(student_positive)  # [batch_size, teacher_dim]
            
            # Normalize projected embeddings: p_hat_k,b^(i) = p_k^(i)(x_b) / ||p_k^(i)(x_b)||
            projected_query = torch.nn.functional.normalize(projected_query, p=2, dim=-1)
            projected_positive = torch.nn.functional.normalize(projected_positive, p=2, dim=-1)
            
            # Compute InfoNCE loss for query-positive pairs
            # For each sample b, we compute the InfoNCE loss
            
            # Calculate similarity scores between projected student and teacher
            # For query: <p_hat_query_b, t_positive_b> (positive pair)
            # Negatives: <p_hat_query_b, t_positive_k'> for k' != b
            
            # Similarity matrix: [batch_size, batch_size]
            # scores[b, k'] = <projected_query[b], teacher_positive[k']>
            similarity_matrix = torch.matmul(projected_query, teacher_positive.t())  # [B, B]
            
            # Apply temperature scaling
            temperature = self.temperature
            similarity_matrix = similarity_matrix / temperature
            
            # Compute InfoNCE loss
            # For each sample b, the positive is at diagonal position [b, b]
            # Loss = -log( exp(<p_hat_b, t_b>) / sum_{k'=1}^B exp(<p_hat_b, t_k'>) )
            
            labels = torch.arange(batch_size, device=similarity_matrix.device)
            loss_query = torch.nn.functional.cross_entropy(similarity_matrix, labels)
            
            # Similarly for positive-query pairs (symmetric distillation)
            similarity_matrix_reverse = torch.matmul(projected_positive, teacher_query.t()) / temperature
            loss_positive = torch.nn.functional.cross_entropy(similarity_matrix_reverse, labels)
            
            # Average the two directions
            infonce_loss = (loss_query + loss_positive) / 2.0
            
            # Weight by relative importance
            weighted_loss = weight * infonce_loss
            total_loss += weighted_loss
        
        # Average over all dimensions (already weighted)
        # The formula in the paper has 1/B factor, but cross_entropy already averages over batch
        # So we just need to normalize by the number of dimensions processed
        if len(dim_indices) > 0:
            total_loss = total_loss / len(dim_indices)
        
        return total_loss


    def record_logging_output(self, logging_output, batch_denom, content):
        """
        Record metrics like loss and accuracy for logging, handling distributed training.
        Enhanced version that handles additional metrics from CKA loss
        """
        import torch.distributed as dist
        
        for k, v in content.items():
            if k == "correct":
                # Sum the correct counts across processes
                record_v = v.clone()
                if dist.is_initialized():
                    dist.all_reduce(record_v, dist.ReduceOp.SUM)
                record_v = record_v.item()
            else:
                # Normalize loss by batch_denom and average across processes
                if isinstance(v, torch.Tensor):
                    record_v = v / batch_denom
                    if dist.is_initialized():
                        dist.all_reduce(record_v, dist.ReduceOp.SUM)
                        record_v = record_v.item() / dist.get_world_size()
                    else:
                        record_v = record_v.item()
                else:
                    record_v = v / batch_denom
                    
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
                
        return logging_output
