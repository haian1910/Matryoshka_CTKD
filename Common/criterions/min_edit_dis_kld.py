import logging
import torch
import torch.nn.functional as F
import numpy as np
import editdistance
from typing import Dict, List, Tuple, Sequence
from .info_NCE import InfoNCELoss

class MinEditDisForwardKLD(InfoNCELoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature if hasattr(args, 'kd_temperature') else 1.0
        
        # Special token mapping will be built dynamically
        self.specTok_mapper = {}
        self.blending_model_special_token = "▁"  # For student (BERT uses ##, LLaMA uses ▁)
        self.base_model_special_token = "▁"      # For teacher
        
        # Projector will be initialized on first forward pass
        self.proj_s2t = None
    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom,
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        
        tokenizer_student = distiller.student_tokenizer
        tokenizer_teacher = distiller.teacher_tokenizer
        
        # Build special token mapper if not already built
        if not self.specTok_mapper:
            self._build_special_token_mapper(tokenizer_student, tokenizer_teacher)
        
        # Get student query embeddings and hidden states
        query_outputs = model(
            input_data['query_input_ids'],
            attention_mask=input_data['query_attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        query_embeddings = distiller.get_embeddings(
            model,
            input_data['query_input_ids'],
            input_data['query_attention_mask']
        )
        query_hidden_states = query_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Get student positive embeddings and hidden states
        positive_outputs = model(
            input_data['positive_input_ids'],
            attention_mask=input_data['positive_attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        positive_embeddings = distiller.get_embeddings(
            model,
            input_data['positive_input_ids'],
            input_data['positive_attention_mask']
        )
        positive_hidden_states = positive_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
        
        log = {}
        
        # Compute InfoNCE loss (task loss)
        task_loss, correct = self.compute_infonce_loss(query_embeddings, positive_embeddings)
        log["task_loss"] = task_loss
        log["correct"] = correct
        
        # Get teacher embeddings and hidden states
        with torch.no_grad():
            teacher_model.eval()
            
            # Teacher query
            teacher_query_outputs = teacher_model(
                input_data['teacher_query_input_ids'],
                attention_mask=input_data['teacher_query_attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
            teacher_query_embeddings = distiller.get_embeddings(
                teacher_model,
                input_data['teacher_query_input_ids'],
                input_data['teacher_query_attention_mask']
            )
            teacher_query_hidden_states = teacher_query_outputs.hidden_states[-1]
            
            # Teacher positive
            teacher_positive_outputs = teacher_model(
                input_data['teacher_positive_input_ids'],
                attention_mask=input_data['teacher_positive_attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
            teacher_positive_embeddings = distiller.get_embeddings(
                teacher_model,
                input_data['teacher_positive_input_ids'],
                input_data['teacher_positive_attention_mask']
            )
            teacher_positive_hidden_states = teacher_positive_outputs.hidden_states[-1]
            
            # Normalize teacher embeddings
            teacher_query_embeddings = F.normalize(teacher_query_embeddings, p=2, dim=-1)
            teacher_positive_embeddings = F.normalize(teacher_positive_embeddings, p=2, dim=-1)
        
        # Compute KD loss using DTW-aligned hidden states
        kd_loss, log = self.compute_dual_branch_kd_loss(
            stu_query_hidden=query_hidden_states,
            stu_pos_hidden=positive_hidden_states,
            tea_query_hidden=teacher_query_hidden_states,
            tea_pos_hidden=teacher_positive_hidden_states,
            stu_query_emb=query_embeddings,
            stu_pos_emb=positive_embeddings,
            tea_query_emb=teacher_query_embeddings,
            tea_pos_emb=teacher_positive_embeddings,
            input_data=input_data,
            tokenizer_student=tokenizer_student,
            tokenizer_teacher=tokenizer_teacher,
            log=log
        )
        
        print("min_ed_kd_loss:", kd_loss.item())
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * task_loss + self.kd_rate * kd_loss
        log["loss"] = loss
        
        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss, logging_output
    
    def _build_special_token_mapper(self, tok_student, tok_teacher):
        """Build mapping between student and teacher special tokens"""
        if tok_student.cls_token and tok_teacher.bos_token:
            self.specTok_mapper[tok_student.cls_token] = tok_teacher.bos_token
        if tok_student.sep_token and tok_teacher.eos_token:
            self.specTok_mapper[tok_student.sep_token] = tok_teacher.eos_token
        if tok_student.pad_token and tok_teacher.pad_token:
            self.specTok_mapper[tok_student.pad_token] = tok_teacher.pad_token
        if tok_student.unk_token and tok_teacher.unk_token:
            self.specTok_mapper[tok_student.unk_token] = tok_teacher.unk_token
        if hasattr(tok_student, 'mask_token') and tok_student.mask_token and \
           hasattr(tok_teacher, 'mask_token') and tok_teacher.mask_token:
            self.specTok_mapper[tok_student.mask_token] = tok_teacher.mask_token
    
    def dist_fn(self, a, b):
        """Calculate edit distance between two tokens"""
        # Special tokens have 0 distance
        if a in self.specTok_mapper and b in self.specTok_mapper.values():
            return 0.0
        if b in self.specTok_mapper and a in self.specTok_mapper.values():
            return 0.0
        
        # Remove special characters and spaces
        aa = a.replace(self.blending_model_special_token, "").replace(" ", "")
        bb = b.replace(self.base_model_special_token, "").replace(" ", "")
        
        if len(aa) == 0 and len(bb) == 0:
            return 0.0
        
        dist = editdistance.eval(aa, bb)
        dist = dist / (len(aa) + len(bb))
        return dist
    
    def dtw(self, series_1, series_2, norm_func):
        """Dynamic Time Warping alignment between two token sequences"""
        matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
        matrix[0, :] = np.inf
        matrix[:, 0] = np.inf
        matrix[0, 0] = 0
        
        # Fill DTW matrix
        for i, vec1 in enumerate(series_1):
            for j, vec2 in enumerate(series_2):
                cost = norm_func(vec1, vec2)
                matrix[i + 1, j + 1] = cost + min(
                    matrix[i, j + 1],     # up
                    matrix[i + 1, j],     # left
                    matrix[i, j]          # diagonal
                )
        
        matrix = matrix[1:, 1:]
        i = matrix.shape[0] - 1
        j = matrix.shape[1] - 1
        
        # Backtrack to find alignment path
        matches = []
        mappings_series_1 = [list() for _ in range(matrix.shape[0])]
        mappings_series_2 = [list() for _ in range(matrix.shape[1])]
        
        while i > 0 or j > 0:
            matches.append((i, j))
            mappings_series_1[i].append(j)
            mappings_series_2[j].append(i)
            
            option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
            option_up = matrix[i - 1, j] if i > 0 else np.inf
            option_left = matrix[i, j - 1] if j > 0 else np.inf
            
            move = np.argmin([option_diag, option_up, option_left])
            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
        
        matches.append((0, 0))
        mappings_series_1[0].append(0)
        mappings_series_2[0].append(0)
        matches.reverse()
        
        for mp in mappings_series_1:
            mp.reverse()
        for mp in mappings_series_2:
            mp.reverse()
        
        return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix
    
    def align_by_one_one(
        self,
        base_vals: torch.Tensor,            # [L_base, d_base]   (teacher)
        blend_vals: torch.Tensor,           # [L_blend, d_blend] (student)
        path: Sequence[Tuple[int, int]],    
        base_tokens: List[str],             
        blend_tokens: List[str],           
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align teacher and student hidden states based on DTW path.
        Uses edit distance to select best token in runs (vertical/horizontal).
        """
        A_base, A_blend = [], []
        k = 0
        P = len(path)
        
        while k < P:
            i0, j0 = path[k]
            if k == P - 1:
                A_base.append(base_vals[i0])
                A_blend.append(blend_vals[j0])
                break
            
            i1, j1 = path[k + 1]
            di, dj = i1 - i0, j1 - j0
            
            # Vertical run: multiple base (teacher) -> 1 blend (student)
            if dj == 0 and di == 1:
                i_run = [i0]
                j_fix = j0
                kk = k + 1
                while kk < P and path[kk][1] == j_fix and path[kk][0] == i_run[-1] + 1:
                    i_run.append(path[kk][0])
                    kk += 1
                
                bj = blend_tokens[j_fix]
                best_i = min(i_run, key=lambda ii: editdistance.eval(base_tokens[ii], bj))
                A_base.append(base_vals[best_i])
                A_blend.append(blend_vals[j_fix])
                k = kk
                continue
            
            # Horizontal run: 1 base (teacher) -> multiple blend (student)
            if di == 0 and dj == 1:
                j_run = [j0]
                i_fix = i0
                kk = k + 1
                while kk < P and path[kk][0] == i_fix and path[kk][1] == j_run[-1] + 1:
                    j_run.append(path[kk][1])
                    kk += 1
                
                bi = base_tokens[i_fix]
                best_j = min(j_run, key=lambda jj: editdistance.eval(bi, blend_tokens[jj]))
                A_base.append(base_vals[i_fix])
                A_blend.append(blend_vals[best_j])
                k = kk
                continue
            
            # Diagonal: 1-1 alignment
            A_base.append(base_vals[i0])
            A_blend.append(blend_vals[j0])
            k += 1
        
        A_base = torch.stack(A_base, dim=0)
        A_blend = torch.stack(A_blend, dim=0)
        return A_base, A_blend
    
    def compute_dual_branch_kd_loss(
        self,
        stu_query_hidden: torch.Tensor,  # [batch_size, seq_len_s, hidden_dim_s]
        stu_pos_hidden: torch.Tensor,
        tea_query_hidden: torch.Tensor,  # [batch_size, seq_len_t, hidden_dim_t]
        tea_pos_hidden: torch.Tensor,
        stu_query_emb: torch.Tensor,     # [batch_size, hidden_dim_s]
        stu_pos_emb: torch.Tensor,
        tea_query_emb: torch.Tensor,     # [batch_size, hidden_dim_t]
        tea_pos_emb: torch.Tensor,
        input_data: Dict,
        tokenizer_student,
        tokenizer_teacher,
        log: Dict
    ):
        """
        Compute KD loss for both query and positive branches using DTW-aligned hidden states.
        """
        device_s = stu_query_hidden.device
        batch_size = stu_query_hidden.size(0)
        
        # Initialize projector if needed
        if self.proj_s2t is None:
            d_s = stu_query_hidden.size(-1)
            d_t = tea_query_hidden.size(-1)
            # Initialize with the same dtype as student hidden states
            self.proj_s2t = torch.nn.Linear(d_s, d_t, bias=False).to(device_s, dtype=stu_query_hidden.dtype)
        
        # Get projector dtype for consistency
        proj_dtype = self.proj_s2t.weight.dtype
        
        # ========== Query Branch ==========
        kd_loss_query = self.compute_branch_kd_loss(
            S_last=stu_query_hidden,
            T_last=tea_query_hidden,
            input_ids=input_data["query_input_ids"],
            teacher_input_ids=input_data["teacher_query_input_ids"],
            attention_mask=input_data["query_attention_mask"],
            teacher_attention_mask=input_data["teacher_query_attention_mask"],
            special_tokens_mask=input_data.get("query_special_tokens_mask"),
            teacher_special_tokens_mask=input_data.get("teacher_query_special_tokens_mask"),
            tokenizer_student=tokenizer_student,
            tokenizer_teacher=tokenizer_teacher
        )
        
        # ========== Positive Branch ==========
        kd_loss_positive = self.compute_branch_kd_loss(
            S_last=stu_pos_hidden,
            T_last=tea_pos_hidden,
            input_ids=input_data["positive_input_ids"],
            teacher_input_ids=input_data["teacher_positive_input_ids"],
            attention_mask=input_data["positive_attention_mask"],
            teacher_attention_mask=input_data["teacher_positive_attention_mask"],
            special_tokens_mask=input_data.get("positive_special_tokens_mask"),
            teacher_special_tokens_mask=input_data.get("teacher_positive_special_tokens_mask"),
            tokenizer_student=tokenizer_student,
            tokenizer_teacher=tokenizer_teacher
        )
        
        # ========== Embedding-level KD (Optional) ==========
        # Match student embeddings with teacher embeddings using MSE
        # Ensure all tensors have consistent dtype
        stu_query_emb_proj = self.proj_s2t(stu_query_emb.to(proj_dtype))
        tea_query_emb_target = tea_query_emb.detach().to(proj_dtype)
        
        stu_pos_emb_proj = self.proj_s2t(stu_pos_emb.to(proj_dtype))
        tea_pos_emb_target = tea_pos_emb.detach().to(proj_dtype)
        
        emb_kd_loss = F.mse_loss(stu_query_emb_proj, tea_query_emb_target) + \
                      F.mse_loss(stu_pos_emb_proj, tea_pos_emb_target)
        
        # Combine all KD losses
        kd_loss = kd_loss_query + kd_loss_positive + emb_kd_loss
        
        # Logging
        log["kd_loss_query"] = kd_loss_query
        log["kd_loss_positive"] = kd_loss_positive
        log["emb_kd_loss"] = emb_kd_loss
        log["kd_loss"] = kd_loss
        
        return kd_loss, log
    
    def compute_branch_kd_loss(
        self,
        S_last: torch.Tensor,  # [batch_size, seq_len_student, hidden_dim_student]
        T_last: torch.Tensor,  # [batch_size, seq_len_teacher, hidden_dim_teacher]
        input_ids: torch.Tensor,
        teacher_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
        special_tokens_mask,
        teacher_special_tokens_mask,
        tokenizer_student,
        tokenizer_teacher
    ):
        """
        Compute KD loss for a single branch (query or positive) using DTW alignment.
        """
        device_s = S_last.device
        batch_size = S_last.size(0)
        
        # Create masks to filter out padding and special tokens
        keep_s = attention_mask.bool()
        keep_t = teacher_attention_mask.bool()
        
        # Filter special tokens if masks are provided
        if special_tokens_mask is not None:
            keep_s = keep_s & (~special_tokens_mask.bool())
        if teacher_special_tokens_mask is not None:
            keep_t = keep_t & (~teacher_special_tokens_mask.bool())
        
        # Convert input_ids to tokens (on CPU for efficiency)
        input_ids_cpu = input_ids.cpu()
        teacher_input_ids_cpu = teacher_input_ids.cpu()
        
        stu_tokens = [
            tokenizer_student.convert_ids_to_tokens(x.tolist(), skip_special_tokens=False)
            for x in input_ids_cpu
        ]
        tea_tokens = [
            tokenizer_teacher.convert_ids_to_tokens(x.tolist(), skip_special_tokens=False)
            for x in teacher_input_ids_cpu
        ]
        
        # Compute KD loss per sample in batch
        loss_kd_sum = 0.0
        denom = 0
        
        for i in range(batch_size):
            # Extract valid tokens (non-padding, non-special)
            S_i = S_last[i][keep_s[i]]  # [valid_len_s, hidden_dim_s]
            T_i = T_last[i][keep_t[i]]  # [valid_len_t, hidden_dim_t]
            
            if S_i.numel() == 0 or T_i.numel() == 0:
                continue
            
            # Get corresponding tokens
            keep_s_mask = keep_s[i].detach().cpu().tolist()
            keep_t_mask = keep_t[i].detach().cpu().tolist()
            
            s_tok_i = [tok for tok, m in zip(stu_tokens[i], keep_s_mask) if m]
            t_tok_i = [tok for tok, m in zip(tea_tokens[i], keep_t_mask) if m]
            
            # Perform DTW alignment
            matches, _, _, _, _ = self.dtw(
                series_1=t_tok_i,
                series_2=s_tok_i,
                norm_func=self.dist_fn
            )
            
            # Align hidden states based on DTW path
            A_t, A_s = self.align_by_one_one(
                base_vals=T_i,
                blend_vals=S_i,
                base_tokens=t_tok_i,
                blend_tokens=s_tok_i,
                path=matches
            )
            
            # Project student to teacher dimension
            S_proj = self.proj_s2t(A_s.to(self.proj_s2t.weight.dtype))  # [M, hidden_dim_t]
            
            # Compute MSE loss
            loss_kd_sum += F.mse_loss(S_proj, A_t.to(S_proj.dtype), reduction="sum")
            denom += A_t.numel()
            
            # Clean up
            del S_proj, A_t, A_s
        
        # Average loss
        loss_kd = loss_kd_sum / max(denom, 1)
        
        return loss_kd
