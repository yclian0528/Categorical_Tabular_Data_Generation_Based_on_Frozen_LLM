import os
import numpy as np
import pandas as pd
from src.utils.config_loader import load_config, set_seed
from src.data.data_preprocessor import DataPreprocessor

def main():
    """Prepare baseline numeric synthetic data for categorical generation."""
    cfg = load_config()

    datasets = cfg.experiment_targets.datasets
    num_gen_models = cfg.experiment_targets.num_gen_models
    
    src_root = cfg.paths.syn_raw_source_root
    tgt_root = cfg.paths.syn_numeric_root
    processed_root = cfg.paths.processed_data_root

    for current_seed in cfg.project.seeds:
        for model_name in num_gen_models:
            for data_name in datasets:
                print(f"[INFO] Preprocessing synthetic data: model={model_name}, dataset={data_name}, seed={current_seed}")

                set_seed(current_seed)
                
                src_path = os.path.join(src_root, model_name, "syn_data", f"seed_{current_seed}", data_name, f"syn_{data_name}.csv")
                output_dir = os.path.join(tgt_root, model_name, "syn_data", f"seed_{current_seed}", data_name)
                tgt_path = os.path.join(output_dir, f"syn_{data_name}.csv")

                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"Synthetic source file not found: {src_path}")

                os.makedirs(output_dir, exist_ok=True)

                df = pd.read_csv(src_path).drop(columns=["__row_id__", "text"], axis=1, errors="ignore")
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                df = df[num_cols]
                
                df.to_csv(tgt_path, index=False)

                preprocessor = DataPreprocessor(
                    data_path=tgt_path,
                    output_dir=output_dir,
                    artifacts_dir=os.path.join(processed_root, data_name, f"seed_{current_seed}", "artifacts"),
                    random_state=current_seed,
                    dump_label_encoders=False,
                    shuffle_num_cols_in_text=cfg.preprocessing.shuffle_num_cols_in_text
                )
                
                preprocessor.preprocess(mode="inference")

    print("[DONE] Synthetic preprocessing completed.")

if __name__ == "__main__":
    main()
