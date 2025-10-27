import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
from utils import log_rank
from typing import Optional
from transformers import AutoTokenizer

class DistillDataset(Dataset):
    def __init__(
        self,
        args,
        student_tokenizer: AutoTokenizer,
        teacher_tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.args = args
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = args.max_length

        self.dataset = self._load_and_process_data()

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_process_data(self):
        dataset = []
        # Load the single combined dataset file
        path = os.path.join(self.args.data_dir, "train.csv")

        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'query' not in df.columns or 'positive' not in df.columns:
                raise ValueError(f"CSV file {path} must contain 'query' and 'positive' columns")
            
            log_rank(f"Processing contrastive dataset with {len(df)} pairs...")  
            
            for _, row in tqdm(df.iterrows(), total=len(df), disable=(dist.get_rank() != 0)):
                # Tokenize query
                student_query_input_ids = self.student_tokenizer.encode(
                    row['query'], 
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True
                )
                
                # Tokenize positive
                student_positive_input_ids = self.student_tokenizer.encode(
                    row['positive'], 
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True
                )
                
                tokenized_data = {
                    "student_query_input_ids": student_query_input_ids,
                    "student_positive_input_ids": student_positive_input_ids,
                }
        
                # If teacher tokenizer is provided, tokenize for teacher as well
                if self.teacher_tokenizer:
                    teacher_query_input_ids = self.teacher_tokenizer.encode(
                        row['query'],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True
                    )
                    teacher_positive_input_ids = self.teacher_tokenizer.encode(
                        row['positive'],
                        add_special_tokens=True,
                        max_length=self.max_length,
                        truncation=True
                    )
                    tokenized_data["teacher_query_input_ids"] = teacher_query_input_ids
                    tokenized_data["teacher_positive_input_ids"] = teacher_positive_input_ids

                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
        
    def _process_contrastive(self, i, samp, model_data):
        """Process query and positive pairs for contrastive learning"""
        # Process student query
        query_ids = np.array(samp["student_query_input_ids"])
        query_len = len(query_ids)
        model_data["query_input_ids"][i][:query_len] = torch.tensor(query_ids, dtype=torch.long)
        model_data["query_attention_mask"][i][:query_len] = 1.0
        
        # Process student positive
        positive_ids = np.array(samp["student_positive_input_ids"])
        positive_len = len(positive_ids)
        model_data["positive_input_ids"][i][:positive_len] = torch.tensor(positive_ids, dtype=torch.long)
        model_data["positive_attention_mask"][i][:positive_len] = 1.0

        # Process teacher if available
        if "teacher_query_input_ids" in samp:
            t_query_ids = np.array(samp["teacher_query_input_ids"])
            t_query_len = len(t_query_ids)
            model_data["teacher_query_input_ids"][i][:t_query_len] = torch.tensor(t_query_ids, dtype=torch.long)
            model_data["teacher_query_attention_mask"][i][:t_query_len] = 1.0
            
            t_positive_ids = np.array(samp["teacher_positive_input_ids"])
            t_positive_len = len(t_positive_ids)
            model_data["teacher_positive_input_ids"][i][:t_positive_len] = torch.tensor(t_positive_ids, dtype=torch.long)
            model_data["teacher_positive_attention_mask"][i][:t_positive_len] = 1.0

    def move_to_device(self, data, device):
        """Move all tensors in data dict to device"""
        for k in data:
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length

        student_pad_token_id = self.student_tokenizer.pad_token_id
        if student_pad_token_id is None:
            student_pad_token_id = 0
        
        # Initialize model data for student query and positive
        model_data = {
            "query_input_ids": torch.ones(bs, max_length, dtype=torch.long) * student_pad_token_id,
            "query_attention_mask": torch.zeros(bs, max_length),
            "positive_input_ids": torch.ones(bs, max_length, dtype=torch.long) * student_pad_token_id,
            "positive_attention_mask": torch.zeros(bs, max_length),
        }

        # Add teacher data if teacher tokenizer exists
        if self.teacher_tokenizer:
            teacher_pad_token_id = self.teacher_tokenizer.pad_token_id
            if teacher_pad_token_id is None:
                teacher_pad_token_id = 0
            model_data.update({
                "teacher_query_input_ids": torch.ones(bs, max_length, dtype=torch.long) * teacher_pad_token_id,
                "teacher_query_attention_mask": torch.zeros(bs, max_length),
                "teacher_positive_input_ids": torch.ones(bs, max_length, dtype=torch.long) * teacher_pad_token_id,
                "teacher_positive_attention_mask": torch.zeros(bs, max_length),
            })

        # Process each sample
        for i, samp in enumerate(samples):
            self._process_contrastive(i, samp, model_data)
        
        return model_data