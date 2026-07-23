import os
import gc
import json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from src.utils.config_loader import load_config, set_seed
from src.data.multi_task_csv_dataset import MultiTaskCsvDataset

def infer_tasks_and_class_dims(ori_data, excluded_cols=["text"]):
    """Infer numeric columns, categorical tasks, and class counts."""
    df = pd.read_csv(ori_data)
    excluded_cols = set(excluded_cols or [])

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    task_cols = [c for c in df.select_dtypes(exclude=["number"]).columns.tolist() if c not in excluded_cols]

    if "__row_id__" in num_cols: num_cols.remove("__row_id__")
    if not task_cols:
        raise ValueError(f"No task columns found: {ori_data}")

    class_dims = {c: df[c].nunique() for c in task_cols}
    return task_cols, class_dims, num_cols

class OfflineFeatureExtractor:
    """Extract frozen-backbone features for seed-specific train/valid splits."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.base_output_dir = cfg.paths.feature_extract_root
        self.device = cfg.project.device
        os.makedirs(self.base_output_dir, exist_ok=True)

    def _get_model_config(self, model_path):
        """Infer dtype and pooling strategy from config and model name."""
        model_name = model_path.lower()
        
        if self.cfg.extraction.auto_dtype:
            if any(x in model_name for x in ["qwen", "llama", "mistral", "7b", "8b"]):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = torch.float32

        if self.cfg.extraction.auto_pooling:
            if any(x in model_name for x in ["bert", "roberta", "deberta", "modernbert", "bge"]):
                pooling_type = "cls"
            elif any(x in model_name for x in ["qwen", "llama"]):
                pooling_type = "last"
            else:
                pooling_type = "mean"
        else:
            pooling_type = "mean"

        return dtype, pooling_type

    @torch.no_grad()
    def run(self, backbone_list, dataset_list, current_seed):
        batch_size = self.cfg.extraction.batch_size
        max_length = self.cfg.extraction.max_length

        for model_info in backbone_list:
            model_path = model_info["path"]
            model_display_name = model_info["name"]
            
            dtype, pooling_type = self._get_model_config(model_path)
            
            print(f"[INFO] Extracting features: backbone={model_display_name}, seed={current_seed}")

            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModel.from_pretrained(
                model_path, local_files_only=True, torch_dtype=dtype
            ).to(self.device).eval()

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            for ds_info in dataset_list:

                output_path = os.path.join(self.base_output_dir, f"seed_{current_seed}", model_display_name, ds_info["name"])
                os.makedirs(output_path, exist_ok=True)

                for split in ["train", "valid"]:
                    save_file = os.path.join(output_path, f"{split}_features.pt")
                    print(f"[INFO] Processing split: dataset={ds_info['name']}, split={split}")
                    
                    dataset = MultiTaskCsvDataset(
                        ds_info[split], tokenizer, 
                        target_cols=ds_info["tasks"], 
                        num_cols=ds_info["num_cols"],
                        max_length=max_length
                    )
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                    all_features, all_raw_nums = [], []
                    all_labels = {c: [] for c in ds_info["tasks"]}

                    for batch in tqdm(loader, desc=f"  {split}"):
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        last_hidden = outputs.last_hidden_state

                        if pooling_type == "cls":
                            sequence_repr = last_hidden[:, 0, :].float()
                        elif pooling_type == "last":
                            sequence_lengths = attention_mask.sum(dim=1) - 1
                            batch_size = last_hidden.shape[0]
                            sequence_repr = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths].float()
                        else:
                            mask = attention_mask.unsqueeze(-1).float()
                            sequence_repr = (last_hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

                        sequence_repr = torch.nan_to_num(sequence_repr, nan=0.0, posinf=10.0, neginf=-10.0)

                        is_embedding_model = any(name in model_display_name.lower() for name in ['bge', 'e5', 'embedding'])
                        if is_embedding_model:
                            sequence_repr = torch.nn.functional.normalize(sequence_repr, p=2, dim=-1)
                        
                        all_features.append(sequence_repr.cpu())
                        if "num_values" in batch:
                            all_raw_nums.append(batch["num_values"].float().cpu())
                        for c in ds_info["tasks"]:
                            all_labels[c].append(batch[c].clone().cpu())

                    save_dict = {
                        "features": torch.cat(all_features, dim=0),
                        "labels": {c: torch.cat(all_labels[c], dim=0) for c in ds_info["tasks"]},
                        "config": {
                            "backbone": model_display_name,
                            "pooling": pooling_type,
                            "hidden_size": model.config.hidden_size,
                            "raw_num_dim": all_raw_nums[0].shape[1] if all_raw_nums else 0
                        }
                    }
                    if all_raw_nums:
                        save_dict["raw_nums"] = torch.cat(all_raw_nums, dim=0)

                    torch.save(save_dict, save_file)
                
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    cfg = load_config()
    
    my_backbones = [
        {"name": bb, "path": os.path.join(cfg.paths.offline_models_dir, bb)} 
        for bb in cfg.experiment_targets.backbones
    ]

    for current_seed in cfg.project.seeds:
        my_datasets = []
        for name in cfg.experiment_targets.datasets:
            set_seed(current_seed)
            raw_csv_path = os.path.join(cfg.paths.raw_data_dir, f"{name}.csv")
            task_cols, class_dims, num_cols = infer_tasks_and_class_dims(raw_csv_path)
            
            os.makedirs(os.path.join(cfg.paths.processed_data_root, name, "artifacts"), exist_ok=True)
            with open(os.path.join(cfg.paths.processed_data_root, name, "artifacts", "class_dims.json"), "w") as f:
                json.dump(class_dims, f, indent=2)

            my_datasets.append({
                "name": name,
                "tasks": task_cols,
                "num_cols": [f"{c}_scaled" for c in num_cols],
                "train": os.path.join(cfg.paths.processed_data_root, name, f"seed_{current_seed}", "processed", "train.csv"),
                "valid": os.path.join(cfg.paths.processed_data_root, name, f"seed_{current_seed}", "processed", "valid.csv")
            })
    
        extractor = OfflineFeatureExtractor(cfg)
        extractor.run(backbone_list=my_backbones, dataset_list=my_datasets, current_seed=current_seed)
