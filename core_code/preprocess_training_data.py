import os
from src.utils.config_loader import load_config, set_seed
from src.data.data_preprocessor import DataPreprocessor

def main():
    """Preprocess raw datasets into seed-specific train and validation files."""
    cfg = load_config()

    for current_seed in cfg.project.seeds:
        for data_name in cfg.experiment_targets.datasets:
            print(f"[INFO] Preprocessing train data: dataset={data_name}, seed={current_seed}")
            set_seed(current_seed)
            
            input_path = os.path.join(cfg.paths.raw_data_dir, f"{data_name}.csv")
            output_dir = os.path.join(cfg.paths.processed_data_root, data_name, f"seed_{current_seed}", "processed")
            artifacts_dir = os.path.join(cfg.paths.processed_data_root, data_name, f"seed_{current_seed}", "artifacts")

            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Dataset file not found: {input_path}")

            preprocessor = DataPreprocessor(
                data_path=input_path,
                output_dir=output_dir,
                artifacts_dir=artifacts_dir,
                float_format=cfg.preprocessing.float_format,
                random_state=current_seed,
                dump_label_encoders=cfg.preprocessing.dump_label_encoders,
                shuffle_num_cols_in_text=cfg.preprocessing.shuffle_num_cols_in_text
            )

            preprocessor.preprocess(mode="train")
            preprocessor.split_train_valid(train_ratio=cfg.preprocessing.train_ratio)

    print("[DONE] Training preprocessing completed.")

if __name__ == "__main__":
    main()
