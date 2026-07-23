import os
import json
import torch
import pandas as pd
import gc

from src.utils.config_loader import load_config
from src.data.semantic_categorical_auto_encoder import SemanticCategoricalEncoder

def infer_tasks_and_class_dims(ori_data, excluded_cols=["text"]):
    """Infer categorical tasks and class counts from a raw CSV file."""
    df = pd.read_csv(ori_data)
    excluded_cols = set(excluded_cols or [])
    
    task_cols = [c for c in df.select_dtypes(exclude=["number"]).columns.tolist() if c not in excluded_cols]
    
    if "__row_id__" in task_cols: task_cols.remove("__row_id__")
    
    if not task_cols:
        raise ValueError(f"No task columns found: {ori_data}")

    class_dims = {c: df[c].nunique() for c in task_cols}
    return task_cols, class_dims

def main():
    """Build global semantic embeddings for categorical labels."""
    cfg = load_config()

    data_names = cfg.experiment_targets.datasets
    backbone_names = cfg.experiment_targets.backbones
    
    for backbone_name in backbone_names:
        print(f"[INFO] Building SCE embeddings: backbone={backbone_name}")
        
        model_path = os.path.join(cfg.paths.offline_models_dir, backbone_name)

        for data_name in data_names:
            print(f"[INFO] Processing dataset: {data_name}")

            encoder = SemanticCategoricalEncoder(
                model_name_or_path=model_path,
                prompt_format=cfg.extraction.prompt_format,
                batch_size=cfg.extraction.batch_size,
                load_mode=False 
            )
            
            raw_csv = os.path.join(cfg.paths.raw_data_dir, f"{data_name}.csv")
            artifacts_dir = os.path.join(cfg.paths.processed_data_root, data_name, "artifacts")
            cat_uniq_val_orders_path = os.path.join(artifacts_dir, "cat_uniq_val_orders.json")
            
            save_dir_name = f"{backbone_name}_{cfg.extraction.prompt_id}_sce"
            save_path = os.path.join(artifacts_dir, save_dir_name)

            task_cols, class_dims = infer_tasks_and_class_dims(raw_csv)
            
            os.makedirs(artifacts_dir, exist_ok=True)
            with open(os.path.join(artifacts_dir, "class_dims.json"), "w") as f:
                json.dump(class_dims, f, indent=2)

            df_raw = pd.read_csv(raw_csv)
            
            print(f"[INFO] Building label embeddings: tasks={len(task_cols)}")
            encoder.build_from_dataframe(
                df_raw,
                cat_cols=task_cols,
                class_order_path=cat_uniq_val_orders_path,
                sort_values=True
            )

            encoder.save(save_path)
            print(f"[DONE] Semantic embeddings saved: {save_path}")

            del encoder
            torch.cuda.empty_cache()
            gc.collect()

    print("[DONE] Semantic embedding generation completed.")

if __name__ == "__main__":
    main()
