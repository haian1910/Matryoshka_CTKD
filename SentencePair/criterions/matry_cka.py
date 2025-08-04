import torch
import torch.nn as nn
from .matry_CE import Matryoshka_CE_Loss
from .matry_CE import Matry_CrossEntropyLoss
import math
import random
from typing import List, Optional

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
    Processor for creating Matryoshka-style sub-matrices from hidden states
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

class MATRY_CKA(Matry_CrossEntropyLoss):
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
        self.matryoshka_hidden_weights = getattr(args, 'matryoshka_hidden_weights', None)
        self.n_dims_per_step = getattr(args, 'n_dims_per_step', -1)
        
        # Initialize CKA loss function
        self.cka_loss_fn = CKALoss()
        
        # Initialize Matryoshka hidden state processor
        self.hidden_processor = MatryoshkaHiddenStateProcessor(
            matryoshka_dims=self.matryoshka_hidden_dims,
            matryoshka_weights=self.matryoshka_hidden_weights,
            n_dims_per_step=self.n_dims_per_step
        )
        
        # Relative importance weights for different nesting dimensions (for CE loss)
        self.relative_importance = getattr(args, 'mrl_relative_importance', None)
        if self.relative_importance is None:
            if self.mrl_efficient:
                self.relative_importance = [1.0]
            else:
                num_dims = len(self.nesting_list)
                self.relative_importance = [1.0 / (1 + math.log(i + 1)) for i in range(num_dims)]

        # Initialize Matryoshka loss for CE
        self.matryoshka_loss = Matryoshka_CE_Loss(
            relative_importance=self.relative_importance,
            label_smoothing=self.label_smoothing
        )
        
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        self.distiller = distiller
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        target = output_data["labels"]
        
        # Get student model outputs with hidden states
        model_outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True  # Important: need hidden states for CKA
        )
        
        logits_dict = model_outputs['logits']
        log = {}
        
        # Cross-entropy loss with ground-truth labels (Matryoshka style)
        ce_loss, nll_loss = self.compute_matryoshka_cross_entropy_loss(logits_dict, target)
        
        # Get teacher outputs with hidden states
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        
        # Compute Matryoshka CKA loss across specified layers
        kd_loss = self.compute_matry_cka_loss(
            model_outputs, teacher_outputs, input_data, output_data, distiller
        )
        
        print(f"matry_cka_loss: {kd_loss}")
        print(f"ce_loss: {ce_loss}")
        
        # Combine losses
        total_loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * kd_loss
        
        # Prepare logging
        log = {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "kd_loss": kd_loss,
            "nll_loss": nll_loss
        }
        
        # Compute accuracy for logging
        correct = self.compute_matryoshka_accuracy(logits_dict, target)
        log["correct"] = correct
        
        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return total_loss, logging_output
    
    def compute_matry_cka_loss(
        self, outputs, teacher_outputs, input_data, output_data, distiller
    ):
        """
        Compute Matryoshka CKA loss across specified student layers
        """
        total_cka_loss = 0.0
        num_layers_processed = 0
        
        # Get student and teacher hidden states
        student_hidden_states = outputs.get('hidden_states', None)
        teacher_hidden_states = teacher_outputs.hidden_states
        
        # Process each specified student layer
        for student_layer_idx in self.student_layers_to_process:
            if student_layer_idx >= len(student_hidden_states):
                continue
                
            # Get student hidden state at this layer
            student_hidden = student_hidden_states[student_layer_idx]  # [batch, seq_len, hidden_dim]
            
            # Find corresponding teacher layer (you might want to make this configurable)
            # For now, using a simple mapping - you can customize this
            teacher_layer_idx = min(2*student_layer_idx, len(teacher_hidden_states) - 1)
            
            # Get aligned teacher hidden states
            aligned_teacher_hidden = self.compute_align_matrix_layer_k(
                student_layer_idx, teacher_layer_idx, outputs, teacher_outputs, 
                distiller, input_data
            )
            
            # Apply Matryoshka processing to this layer pair
            layer_cka_loss = self.hidden_processor.process_hidden_states(
                student_hidden, aligned_teacher_hidden, self.cka_loss_fn
            )
            
            total_cka_loss += layer_cka_loss
            num_layers_processed += 1
        
        # Average across processed layers
        if num_layers_processed > 0:
            total_cka_loss = total_cka_loss / num_layers_processed
            
        return total_cka_loss

    def compute_align_matrix_layer_k(self, student_layer_k, teacher_layer_l, outputs, teacher_outputs, distiller, input_data):
        """
        Compute aligned hidden states for student layer k and teacher layer l
        This implements the attention-based alignment mechanism
        """
        # Get hidden states
        student_hidden_states = outputs.get('hidden_states', None)

        stu_hiddens = student_hidden_states[student_layer_k]
        tea_hiddens = teacher_outputs.hidden_states[teacher_layer_l]  # [batch, seq_len, tea_dim]
        
        # Get embedding layers for query/key computation
        stu_embed_tokens = self.get_embedding_layer(distiller.student_model, "student")
        tea_embed_tokens = self.get_embedding_layer(distiller.teacher_model, "teacher")
        
        # Get input embeddings
        stu_input_embeds = stu_embed_tokens(input_data["input_ids"]).detach()
        tea_input_embeds = tea_embed_tokens(input_data["teacher_input_ids"]).detach()
        
        # Normalize teacher embeddings for stability
        norm_tea_input_embeds = tea_input_embeds / (tea_input_embeds.std() + 1e-8)
        
        # Use projector if available, otherwise use hidden states directly
        if hasattr(distiller, 'projectors') and "query" in distiller.projectors:
            stu_q_hiddens = distiller.projectors["query"](stu_hiddens).float()
        else:
            stu_q_hiddens = stu_hiddens.float()
        
        # Use teacher input embeddings as keys
        tea_k_hiddens = norm_tea_input_embeds.float()
        
        # Normalize teacher hidden states for values
        norm_teacher_hiddens = tea_hiddens / (tea_hiddens.std() + 1e-8)
        tea_v_hiddens = norm_teacher_hiddens.float()
        
        # Compute attention alignment: Q * K^T
        # stu_q_hiddens: [batch, seq_len, hidden_dim]
        # tea_k_hiddens: [batch, seq_len, hidden_dim]
        align = torch.matmul(stu_q_hiddens, tea_k_hiddens.transpose(-1, -2))
        
        # Scale by sqrt of dimension
        scale_factor = math.sqrt(tea_k_hiddens.shape[-1])
        align = align / scale_factor
        
        # Apply softmax to get attention weights
        t2s_weight = torch.softmax(align, dim=-1)  # [batch, stu_seq_len, tea_seq_len]
        
        # Apply attention to teacher values
        t2s_hiddens = torch.matmul(t2s_weight, tea_v_hiddens)  # [batch, stu_seq_len, tea_hidden_dim]
        
        # Ensure output is on the same device as student hiddens
        t2s_hiddens = t2s_hiddens.to(stu_hiddens.device, stu_hiddens.dtype)
        
        return t2s_hiddens

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
