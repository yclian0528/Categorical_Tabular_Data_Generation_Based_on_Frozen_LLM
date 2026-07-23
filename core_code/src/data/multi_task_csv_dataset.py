import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils.config_loader import load_config

class MultiTaskCsvDataset(Dataset):
    """Tokenize preprocessed CSV rows and return multi-task labels."""

    def __init__(
        self, 
        path, 
        tokenizer, 
        target_cols=None, 
        num_cols=None, 
        max_length=None
    ):
        cfg = load_config()
        
        self.df = pd.read_csv(path)
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else cfg.extraction.max_length

        if target_cols is None:
            self.target_cols = [
                c for c in self.df.select_dtypes(exclude=["number"]).columns 
                if c not in ["text", "__row_id__"]
            ]
        else:
            self.target_cols = target_cols

        self.num_cols = num_cols if num_cols is not None else []
        
        if not self.target_cols:
            raise ValueError(f"No target columns found: {path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        enc = self.tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}

        for col in self.target_cols:
            item[col] = torch.tensor(int(row[col]), dtype=torch.long)

        if self.num_cols:
            num_values = row[self.num_cols].values.astype(float)
            item["num_values"] = torch.tensor(num_values, dtype=torch.float32)

        return item
