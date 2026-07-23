import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
import json
from src.utils.config_loader import load_config

class SemanticCategoricalEncoder:
    """Build, save, and load semantic embeddings for categorical labels."""

    def __init__(
        self,
        model_name_or_path,
        device=None,
        prompt_format=None,
        batch_size=None,
        load_mode=False,
        torch_dtype=None,
        pooling_type=None,
    ):
        cfg = load_config()
        
        self.model_name_or_path = model_name_or_path
        self.device = device if device is not None else cfg.project.device
        self.prompt_format = prompt_format if prompt_format is not None else cfg.extraction.prompt_format
        self.batch_size = batch_size if batch_size is not None else cfg.extraction.batch_size
        self.max_length = cfg.extraction.max_length
        self.load_mode = load_mode

        auto_dtype, auto_pooling = self._get_auto_config(model_name_or_path)
        
        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        else:
            self.torch_dtype = auto_dtype if cfg.extraction.auto_dtype else torch.float32

        if pooling_type is not None:
            self.pooling_type = pooling_type
        else:
            self.pooling_type = auto_pooling if cfg.extraction.auto_pooling else "mean"

        self.label_list = {}
        self.label_embeddings = {}
        self.label_texts = {}

        if not load_mode:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                local_files_only=True, 
                torch_dtype=self.torch_dtype
            ).to(self.device).eval()
        else:
            self.tokenizer = None
            self.model = None

    def _get_auto_config(self, model_path):
        """Infer dtype and pooling strategy from a model path."""
        m_name = model_path.lower()
        
        if any(x in m_name for x in ["qwen", "llama", "mistral", "7b", "8b"]):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        if any(x in m_name for x in ["bert", "roberta", "deberta", "modernbert", "bge"]):
            pooling_type = "cls"
        elif any(x in m_name for x in ["qwen", "llama"]):
            pooling_type = "last"
        else:
            pooling_type = "mean"

        return dtype, pooling_type

    @torch.no_grad()
    def _encode_texts(self, texts):
        """Encode label prompts into dense vectors."""
        if self.load_mode:
            raise RuntimeError("Encoding is unavailable in load mode.")

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)

            outputs = self.model(**enc)
            hidden = outputs.last_hidden_state

            if self.pooling_type == "cls":
                batch_embeddings = hidden[:, 0, :]
            elif self.pooling_type == "last":
                sequence_lengths = enc["attention_mask"].sum(dim=1) - 1
                batch_size = hidden.shape[0]
                batch_embeddings = hidden[torch.arange(batch_size, device=hidden.device), sequence_lengths]
            else: 
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                batch_embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

            batch_embeddings = batch_embeddings.float()

            all_embeddings.append(batch_embeddings.float().cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=12.0, neginf=-12.0)

        is_embedding_model = any(name in self.model_name_or_path.lower() for name in ['bge', 'e5', 'embedding'])
        if is_embedding_model:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def build_each_column(self, col_name, unique_values):
        """Build label prompts and embeddings for one categorical column."""
        labels = [str(v) for v in unique_values]
        self.label_list[col_name] = labels

        texts, label_to_text = [], {}
        for label in labels:
            text = self.prompt_format.format(col=col_name, val=label)
            texts.append(text)
            label_to_text[label] = text

        self.label_texts[col_name] = label_to_text
        self.label_embeddings[col_name] = self._encode_texts(texts)

    def build_from_dataframe(self, df, cat_cols=None, sort_values=True, class_order_path=None):
        """Build embeddings for categorical columns in a dataframe."""
        if cat_cols is None:
            cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

        class_orders = {}
        if class_order_path and os.path.exists(class_order_path):
            with open(class_order_path, "r", encoding="utf-8") as f:
                class_orders = json.load(f)

        for col in cat_cols:
            if col in class_orders:
                unique_values = class_orders[col]
            else:
                unique_values = df[col].dropna().astype(str).unique()
                if sort_values:
                    unique_values = sorted(unique_values)
            
            self.build_each_column(col, unique_values)

    def _get_safe_name(self, col_name):
        """Return a filesystem-safe column name."""
        return col_name.replace("\\", "_").replace("/", "_")
    
    def save(self, save_dir):
        """Save metadata, label prompts, and embedding arrays."""
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "texts"), exist_ok=True)

        metadata = {
            "model_name_or_path": self.model_name_or_path,
            "prompt_format": self.prompt_format,
            "pooling_type": self.pooling_type,
            "columns": list(self.label_list.keys()),
            "label_list": self.label_list,
        }

        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        for col, emb in self.label_embeddings.items():
            safe_col = self._get_safe_name(col)
            np.save(os.path.join(save_dir, "embeddings", f"{safe_col}.npy"), emb.numpy())

        for col, mapping in self.label_texts.items():
            safe_col = self._get_safe_name(col)
            with open(os.path.join(save_dir, "texts", f"{safe_col}.json"), "w", encoding="utf-8") as f:
                json.dump(mapping, f, indent=4)

    def load(self, save_dir):
        """Load previously saved semantic embeddings."""
        with open(os.path.join(save_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.prompt_format = metadata["prompt_format"]
        self.label_list = metadata["label_list"]
        self.pooling_type = metadata.get("pooling_type", "mean")

        cols_to_load = metadata.get("columns", list(self.label_list.keys()))

        for col in cols_to_load:
            safe_col = self._get_safe_name(col)
            col_emb_path = os.path.join(save_dir, "embeddings", f"{safe_col}.npy")
            
            if os.path.exists(col_emb_path):
                arr = np.load(col_emb_path)
                self.label_embeddings[col] = torch.tensor(arr, dtype=torch.float32)
            else:
                raise FileNotFoundError(f"Embedding file not found: {col_emb_path}")
                
            with open(os.path.join(save_dir, "texts", f"{safe_col}.json"), "r", encoding="utf-8") as f:
                self.label_texts[col] = json.load(f)
                
    def get_embeddings(self, col_name):
        return self.label_embeddings[col_name]

    def get_label_list(self, col_name):
        return self.label_list[col_name]

    def get_label_texts(self, col_name):
        return self.label_texts[col_name]
