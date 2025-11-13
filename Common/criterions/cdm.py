import logging
import torch
import torch.nn.functional as F
import numpy as np
import editdistance
from typing import Dict, List, Tuple, Sequence
from .info_NCE import InfoNCELoss


def calculate_weight(logits):
    """Calculate weight factor based on logits entropy/uncertainty"""
    with torch.no_grad():
        logits = logits.float()
        probs = F.softmax(logits.detach(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float, device=logits.device))
        weight = entropy / max_entropy
        return weight.cpu().float().numpy()


class CDM(InfoNCELoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature if hasattr(args, 'kd_temperature') else 1.0
        
        # Special token mapping
        self.specTok_mapper = {}
        self.blending_model_special_token = "▁"
        self.base_model_special_token = "▁"
        
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
        
        # ========== STUDENT - Query Branch ==========
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
        query_hidden_states = query_outputs.hidden_states[-1]
        
        # Get query logits for weight calculation (if available)
        query_logits = query_outputs.logits if hasattr(query_outputs, 'logits') else None
        
        # ========== STUDENT - Positive Branch ==========
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
        positive_hidden_states = positive_outputs.hidden_states[-1]
        
        # Get positive logits for weight calculation (if available)
        positive_logits = positive_outputs.logits if hasattr(positive_outputs, 'logits') else None
        
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
        
        log = {}
        
        # Compute InfoNCE loss (task loss)
        task_loss, correct = self.compute_infonce_loss(query_embeddings, positive_embeddings)
        log["task_loss"] = task_loss
        log["correct"] = correct
        
        # ========== TEACHER ==========
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
            teacher_query_logits = teacher_query_outputs.logits if hasattr(teacher_query_outputs, 'logits') else None
            
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
            teacher_positive_logits = teacher_positive_outputs.logits if hasattr(teacher_positive_outputs, 'logits') else None
            
            # Normalize teacher embeddings
            teacher_query_embeddings = F.normalize(teacher_query_embeddings, p=2, dim=-1)
            teacher_positive_embeddings = F.normalize(teacher_positive_embeddings, p=2, dim=-1)
        
        # Compute CDM KD loss with weighted DTW
        kd_loss, log = self.compute_cdm_kd_loss(
            stu_query_hidden=query_hidden_states,
            stu_pos_hidden=positive_hidden_states,
            tea_query_hidden=teacher_query_hidden_states,
            tea_pos_hidden=teacher_positive_hidden_states,
            stu_query_logits=query_logits,
            stu_pos_logits=positive_logits,
            tea_query_logits=teacher_query_logits,
            tea_pos_logits=teacher_positive_logits,
            stu_query_emb=query_embeddings,
            stu_pos_emb=positive_embeddings,
            tea_query_emb=teacher_query_embeddings,
            tea_pos_emb=teacher_positive_embeddings,
            input_data=input_data,
            tokenizer_student=tokenizer_student,
            tokenizer_teacher=tokenizer_teacher,
            log=log
        )
        
        print("cdm_kd_loss:", kd_loss.item())
        
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
        if a in self.specTok_mapper and b in self.specTok_mapper.values():
            return 0.0
        if b in self.specTok_mapper and a in self.specTok_mapper.values():
            return 0.0
        
        aa = a.replace(self.blending_model_special_token, "").replace(" ", "")
        bb = b.replace(self.base_model_special_token, "").replace(" ", "")
        
        if len(aa) == 0 and len(bb) == 0:
            return 0.0
        
        dist = editdistance.eval(aa, bb)
        dist = dist / (len(aa) + len(bb))
        return dist
    
    def dtw(self, series_1, series_2, series1_factor=None, series2_factor=None, norm_func=None):
        """
        Dynamic Time Warping with optional weighting factors.
        series1_factor, series2_factor: weights for each token position
        """
        matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
        matrix[0, :] = np.inf
        matrix[:, 0] = np.inf
        matrix[0, 0] = 0
        
        if series1_factor is not None and series2_factor is not None:
            # Weighted DTW
            for i, (vec1, fc1) in enumerate(zip(series_1, series1_factor)):
                for j, (vec2, fc2) in enumerate(zip(series_2, series2_factor)):
                    cost = norm_func(vec1, vec2) * fc1 * fc2
                    matrix[i + 1, j + 1] = cost + min(
                        matrix[i, j + 1],
                        matrix[i + 1, j],
                        matrix[i, j]
                    )
        else:
            # Standard DTW
            for i, vec1 in enumerate(series_1):
                for j, vec2 in enumerate(series_2):
                    cost = norm_func(vec1, vec2)
                    matrix[i + 1, j + 1] = cost + min(
                        matrix[i, j + 1],
                        matrix[i + 1, j],
                        matrix[i, j]
                    )
        
        matrix = matrix[1:, 1:]
        i = matrix.shape[0] - 1
        j = matrix.shape[1] - 1
        
        # Backtrack
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
    
    def align_by_path_pool_many(
        self,
        base_vals: torch.Tensor,    # [L_base, d_base] (teacher)
        blend_vals: torch.Tensor,   # [L_blend, d_blend] (student)
        path: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align using mean pooling for runs (vertical/horizontal).
        Returns aligned tensors of shape [M, d].
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
            
            # Vertical run: multiple base -> 1 blend (pool base)
            if dj == 0 and di == 1:
                i_run = [i0]
                j_fix = j0
                kk = k + 1
                while kk < P and path[kk][1] == j_fix and path[kk][0] == i_run[-1] + 1:
                    i_run.append(path[kk][0])
                    kk += 1
                A_base.append(base_vals[i_run].mean(dim=0))
                A_blend.append(blend_vals[j_fix])
                k = kk
                continue
            
            # Horizontal run: 1 base -> multiple blend (pool blend)
            if di == 0 and dj == 1:
                j_run = [j0]
                i_fix = i0
                kk = k + 1
                while kk < P and path[kk][0] == i_fix and path[kk][1] == j_run[-1] + 1:
                    j_run.append(path[kk][1])
                    kk += 1
                A_base.append(base_vals[i_fix])
                A_blend.append(blend_vals[j_run].mean(dim=0))
                k = kk
                continue
            
            # Diagonal: 1-1
            A_base.append(base_vals[i0])
            A_blend.append(blend_vals[j0])
            k += 1
        
        A_base = torch.stack(A_base, dim=0)
        A_blend = torch.stack(A_blend, dim=0)
        return A_base, A_blend
    
    def compute_cdm_kd_loss(
        self,
        stu_query_hidden: torch.Tensor,
        stu_pos_hidden: torch.Tensor,
        tea_query_hidden: torch.Tensor,
        tea_pos_hidden: torch.Tensor,
        stu_query_logits,
        stu_pos_logits,
        tea_query_logits,
        tea_pos_logits,
        stu_query_emb: torch.Tensor,
        stu_pos_emb: torch.Tensor,
        tea_query_emb: torch.Tensor,
        tea_pos_emb: torch.Tensor,
        input_data: Dict,
        tokenizer_student,
        tokenizer_teacher,
        log: Dict
    ):
        """
        Compute CDM KD loss with weighted DTW for both query and positive branches.
        """
        device_s = stu_query_hidden.device
        batch_size = stu_query_hidden.size(0)
        
        # Initialize projector if needed
        if self.proj_s2t is None:
            d_s = stu_query_hidden.size(-1)
            d_t = tea_query_hidden.size(-1)
            # Initialize with the same dtype as student hidden states
            self.proj_s2t = torch.nn.Linear(d_s, d_t, bias=False).to(device_s, dtype=stu_query_hidden.dtype)
        
        # Calculate weight factors from logits (entropy-based weighting)
        # If logits not available, use uniform weights
        if stu_query_logits is not None and tea_query_logits is not None:
            stu_query_weights = calculate_weight(stu_query_logits)
            tea_query_weights = calculate_weight(tea_query_logits)
        else:
            stu_query_weights = None
            tea_query_weights = None
        
        if stu_pos_logits is not None and tea_pos_logits is not None:
            stu_pos_weights = calculate_weight(stu_pos_logits)
            tea_pos_weights = calculate_weight(tea_pos_logits)
        else:
            stu_pos_weights = None
            tea_pos_weights = None
        
        # ========== Query Branch ==========
        kd_loss_query = self.compute_branch_cdm_kd_loss(
            S_last=stu_query_hidden,
            T_last=tea_query_hidden,
            stu_weights=stu_query_weights,
            tea_weights=tea_query_weights,
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
        kd_loss_positive = self.compute_branch_cdm_kd_loss(
            S_last=stu_pos_hidden,
            T_last=tea_pos_hidden,
            stu_weights=stu_pos_weights,
            tea_weights=tea_pos_weights,
            input_ids=input_data["positive_input_ids"],
            teacher_input_ids=input_data["teacher_positive_input_ids"],
            attention_mask=input_data["positive_attention_mask"],
            teacher_attention_mask=input_data["teacher_positive_attention_mask"],
            special_tokens_mask=input_data.get("positive_special_tokens_mask"),
            teacher_special_tokens_mask=input_data.get("teacher_positive_special_tokens_mask"),
            tokenizer_student=tokenizer_student,
            tokenizer_teacher=tokenizer_teacher
        )
        
        # ========== Embedding-level KD ==========
        # Ensure embeddings match projector dtype
        proj_dtype = self.proj_s2t.weight.dtype
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
    
    def compute_branch_cdm_kd_loss(
        self,
        S_last: torch.Tensor,
        T_last: torch.Tensor,
        stu_weights,  # numpy array or None
        tea_weights,  # numpy array or None
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
        Compute CDM KD loss for a single branch using weighted DTW.
        """
        device_s = S_last.device
        batch_size = S_last.size(0)
        
        # Create masks
        keep_s = attention_mask.bool()
        keep_t = teacher_attention_mask.bool()
        
        if special_tokens_mask is not None:
            keep_s = keep_s & (~special_tokens_mask.bool())
        if teacher_special_tokens_mask is not None:
            keep_t = keep_t & (~teacher_special_tokens_mask.bool())
        
        # Convert to tokens
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
        
        # Compute KD loss per sample
        loss_kd_sum = 0.0
        denom = 0
        
        for i in range(batch_size):
            S_i = S_last[i][keep_s[i]]
            T_i = T_last[i][keep_t[i]]
            
            if S_i.numel() == 0 or T_i.numel() == 0:
                continue
            
            # Get tokens
            keep_s_mask = keep_s[i].detach().cpu().tolist()
            keep_t_mask = keep_t[i].detach().cpu().tolist()
            
            s_tok_i = [tok for tok, m in zip(stu_tokens[i], keep_s_mask) if m]
            t_tok_i = [tok for tok, m in zip(tea_tokens[i], keep_t_mask) if m]
            
            # Extract weight factors for this sample
            if stu_weights is not None and tea_weights is not None:
                # Handle different weight dimensions
                if stu_weights.ndim == 0:
                    # Scalar weight for entire batch
                    s_fac_i = [float(stu_weights)] * len(s_tok_i)
                elif stu_weights.ndim == 1:
                    # Per-sample weight
                    s_fac_i = [float(stu_weights[i])] * len(s_tok_i)
                elif stu_weights.ndim == 2:
                    # Per-token weight
                    s_fac_i = stu_weights[i][keep_s_mask].tolist()
                else:
                    s_fac_i = None
                
                if tea_weights.ndim == 0:
                    t_fac_i = [float(tea_weights)] * len(t_tok_i)
                elif tea_weights.ndim == 1:
                    t_fac_i = [float(tea_weights[i])] * len(t_tok_i)
                elif tea_weights.ndim == 2:
                    t_fac_i = tea_weights[i][keep_t_mask].tolist()
                else:
                    t_fac_i = None
            else:
                s_fac_i = None
                t_fac_i = None
            
            # Perform weighted DTW
            matches, _, _, _, _ = self.dtw(
                series_1=t_tok_i,
                series_2=s_tok_i,
                series1_factor=t_fac_i,
                series2_factor=s_fac_i,
                norm_func=self.dist_fn
            )
            
            # Align with pooling
            A_t, A_s = self.align_by_path_pool_many(
                base_vals=T_i,
                blend_vals=S_i,
                path=matches
            )
            
            # Project and compute loss
            S_proj = self.proj_s2t(A_s.to(self.proj_s2t.weight.dtype))
            loss_kd_sum += F.mse_loss(S_proj, A_t.to(S_proj.dtype), reduction="sum")
            denom += A_t.numel()
            
            del S_proj, A_t, A_s
        
        loss_kd = loss_kd_sum / max(denom, 1)
        return loss_kd
