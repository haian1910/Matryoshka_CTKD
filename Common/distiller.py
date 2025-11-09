import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from utils import log_rank
from huggingface_hub import login
import torch.distributed as dist
import os

class Distiller(nn.Module):
    def __init__(self, args, device):
        super(Distiller, self).__init__()
        self.args = args
        self.device = device
        self.student_model, self.student_tokenizer = self.load_student_model()
        
        # Load teacher model for distillation (required for matry_ctkd criterion)
        if hasattr(self.args, 'teacher_model_path') and self.args.teacher_model_path is not None:
            self.teacher_model, self.teacher_tokenizer = self.load_teacher_model()
        else:
            # For matry_ctkd, we always need a teacher model (uses hardcoded LLM2Vec)
            self.teacher_model, self.teacher_tokenizer = self.load_teacher_model()

    @staticmethod
    def add_distiller_args(parser):
        group = parser.add_argument_group("distiller", "distiller configurations")
        group.add_argument("--pooling-method", type=str, default="cls",
                           choices=["cls", "mean", "last"],
                           help='pooling method for embeddings: cls, mean, or last token')
        return parser
    
    def load_tokenizer(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return tokenizer
    
    def mean_pooling(self, last_hidden_state, attention_mask):
        """Mean pooling - take attention mask into account for correct averaging"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_embeddings(self, model, input_ids, attention_mask):
        """Get embeddings from model based on pooling method"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # if self.args.pooling_method == "cls":
        #     # Use [CLS] token embedding (first token)
        #     embeddings = outputs.last_hidden_state[:, 0, :]
        # elif self.args.pooling_method == "mean":
        #     # Use mean pooling
        #     embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        # elif self.args.pooling_method == "last":
        #     # Use last token embedding
        #     sequence_lengths = attention_mask.sum(dim=1) - 1
        #     batch_size = input_ids.shape[0]
        #     embeddings = outputs.last_hidden_state[torch.arange(batch_size), sequence_lengths]
        # else:
        #     raise ValueError(f"Unknown pooling method: {self.args.pooling_method}")
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        return embeddings
    
    def load_student_model(self):
        log_rank("Loading student model...")
    
        if self.args.model_dtype == "fp32":
            self.dtype = torch.float32
        elif self.args.model_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif self.args.model_dtype == "fp16":
            self.dtype = torch.float16
        else:
            raise NotImplementedError(f"Invalid model_dtype for `{self.args.model_dtype}`")

        if self.args.peft is not None:  # for LLM2Vec with LoRA
            if self.args.peft == "lora":
                config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", trust_remote_code=True)
                config.is_model_parallel = False
        
                tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")
                
                if hasattr(config, "n_embed"):
                    self.hidden_size = config.n_embed
                else:
                    self.hidden_size = config.hidden_size
        
                # Load as AutoModel (backbone only, no classification head)
                model = AutoModel.from_pretrained(
                    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                    config=config,
                    device_map=None,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                )

                model.config.pad_token_id = 2
                    
                model = PeftModel.from_pretrained(
                    model,
                    "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                )
                model = model.merge_and_unload()

                model = PeftModel.from_pretrained(
                    model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse"
                )
                model = model.merge_and_unload() 
                
                # Apply new LoRA adapter for fine-tuning
                if self.args.do_train:
                    peft_config = LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,  # Changed from SEQ_CLS
                        inference_mode=(not self.args.do_train),
                        r=self.args.peft_lora_r,
                        lora_alpha=self.args.peft_lora_alpha,
                        lora_dropout=self.args.peft_lora_dropout,
                        target_modules=[
                            "q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"
                        ]
                    )
                    model = get_peft_model(model, peft_config)
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    all_params = sum(p.numel() for p in model.parameters())
                    log_rank(f"Trainable parameters: {trainable_params}/{all_params} ({trainable_params/all_params:.2%})")
            else:
                raise NotImplementedError
        else:  # for BERT
            config = AutoConfig.from_pretrained("bert-base-uncased", trust_remote_code=True)
            config.is_model_parallel = False
    
            tokenizer = self.load_tokenizer("bert-base-uncased")
            
            if hasattr(config, "n_embed"):
                self.hidden_size = config.n_embed
            else:
                self.hidden_size = config.hidden_size
            
            # Load as AutoModel (backbone only)
            model = AutoModel.from_pretrained(
                "bert-base-uncased", 
                config=config, 
                device_map=None, 
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            log_rank(' > number of parameters: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model, tokenizer
    
    def load_teacher_model(self):
        log_rank("Loading teacher model...")
        config = AutoConfig.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            trust_remote_code=True
        )
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")

        if hasattr(config, "n_embed"):
            self.teacher_hidden_size = config.n_embed
        else:
            self.teacher_hidden_size = config.hidden_size

        # Load as AutoModel (backbone only)
        model = AutoModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            config=config,
            device_map=None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        model.config.pad_token_id = 2
        
        teacher_model = PeftModel.from_pretrained(
            model,
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        )    
        
        teacher_model = teacher_model.merge_and_unload()

        teacher_model = PeftModel.from_pretrained(
            teacher_model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised"
        )
        teacher_model = teacher_model.merge_and_unload()

        # if hasattr(self.args, 'teacher_model_path') and self.args.teacher_model_path:
        #     adapter_path = os.path.join(self.args.teacher_model_path, "adapter_model.bin")
        #     fixed_adapter_path = adapter_path + ".fixed"
            
        #     if not os.path.exists(fixed_adapter_path):
        #         if dist.get_rank() == 0:
        #             checkpoint = torch.load(adapter_path)            
        #             fixed_checkpoint = {}
                    
        #             for key, value in checkpoint.items():
        #                 if "lora_A.weight" in key and "default" not in key:
        #                     key = key.replace("lora_A.weight", "lora_A.default.weight")
        #                 if "lora_B.weight" in key and "default" not in key:
        #                     key = key.replace("lora_B.weight", "lora_B.default.weight")
        #                 if "base_model.model.base_model.model" in key:
        #                     key = key.replace("base_model.model.base_model.model", "base_model.model")
                            
        #                 fixed_checkpoint[key] = value
                    
        #             if fixed_checkpoint: 
        #                 torch.save(fixed_checkpoint, fixed_adapter_path)
            
        #     dist.barrier()  
            
        #     teacher_model = PeftModel.from_pretrained(
        #         teacher_model,
        #         self.args.teacher_model_path,
        #         adapter_name="default",
        #         adapter_weights_path=fixed_adapter_path
        #     )

        # Set teacher model to eval mode and freeze parameters
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        return teacher_model, tokenizer

    def forward(self, criterion, batch, logging_output, loss_denom):
        input_data = batch["input_batch"]
        output_data = batch.get("output_batch", None)
        loss, logging_output = criterion(
            self,
            input_data, 
            output_data,
            logging_output,
            loss_denom,
        )
        return loss, logging_output
