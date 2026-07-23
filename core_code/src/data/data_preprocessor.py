import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

import pickle
import json
from src.utils.config_loader import load_config, set_seed

class DataPreprocessor:
    """Prepare tabular data for training and synthetic-data inference."""

    def __init__(
            self,
            data_path,
            output_dir=None,
            artifacts_dir=None,
            float_format=None,
            random_state=None,
            dump_label_encoders=None,
            shuffle_num_cols_in_text=None,
        ):
        cfg = load_config()

        if random_state is None:
            raise ValueError("random_state is required.")
        set_seed(random_state)
        
        self.data_path = data_path
        self.file_name = data_path.split("/")[-1]
        self.df = pd.read_csv(self.data_path)

        self.scaler = StandardScaler()
        self.is_scaler_fitted = False

        self.row_id_col = "__row_id__"
        if self.row_id_col not in self.df.columns:
            self.df[self.row_id_col] = np.arange(len(self.df), dtype=np.int64)

        self.random_state = random_state
        self.float_format = float_format if float_format is not None else cfg.preprocessing.float_format
        self.output_dir = output_dir if output_dir is not None else os.path.join(cfg.paths.processed_data_root, "default_out")
        self.artifacts_dir = artifacts_dir if artifacts_dir is not None else os.path.join(cfg.paths.processed_data_root, "default_artifacts")
        self.dump_label_encoders = dump_label_encoders if dump_label_encoders is not None else cfg.preprocessing.dump_label_encoders
        self.shuffle_num_cols_in_text = shuffle_num_cols_in_text if shuffle_num_cols_in_text is not None else cfg.preprocessing.shuffle_num_cols_in_text

        self.encoders = {}
        self.num_cols, self.cat_cols = self.get_column_types()
        self.scaled_num_cols = [f"{c}_scaled" for c in self.num_cols]

    def get_column_types(self):
        """Return numeric and categorical column names."""
        num_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=["number"]).columns.tolist()

        if "__row_id__" in num_cols: num_cols.remove("__row_id__")
        if "__row_id__" in cat_cols: cat_cols.remove("__row_id__")

        return num_cols, cat_cols

    def _fit_scaler(self, train_df):
        """Fit and persist the scaler on the training split only."""
        if not self.num_cols: return
        self.scaler.fit(train_df[self.num_cols])
        self.is_scaler_fitted = True

        if self.artifacts_dir:
            os.makedirs(self.artifacts_dir, exist_ok=True)
            scaler_path = os.path.join(self.artifacts_dir, "numerical_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            
            with open(os.path.join(self.artifacts_dir, "num_column_map.json"), "w") as f:
                json.dump({"original": self.num_cols, "scaled": self.scaled_num_cols}, f, indent=2)

    def load_scaler(self):
        """Load the scaler produced by training preprocessing."""
        scaler_path = os.path.join(self.artifacts_dir, "numerical_scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            self.is_scaler_fitted = True
        else:
            print(f"[WARN] Scaler not found: {scaler_path}")

    def apply_scaling(self, target_df):
        """Append scaled numeric columns to a dataframe."""
        if not self.num_cols: return target_df
        if not self.is_scaler_fitted:
            self.load_scaler()

        if self.is_scaler_fitted:
            target_df[self.scaled_num_cols] = self.scaler.transform(target_df[self.num_cols])
        return target_df

    def split_train_valid(self, train_ratio=None):
        """Create deterministic train/validation splits and scale numeric columns."""
        if train_ratio is None:
            cfg = load_config()
            train_ratio = cfg.preprocessing.train_ratio

        rng = np.random.RandomState(self.random_state)
        idx = self.df[self.row_id_col].to_numpy()
        rng.shuffle(idx)

        n_train = int(len(idx) * train_ratio)
        train_ids = set(idx[:n_train])

        train_df = self.df[self.df[self.row_id_col].isin(train_ids)].copy()
        valid_df = self.df[~self.df[self.row_id_col].isin(train_ids)].copy()

        self._fit_scaler(train_df)
        train_df = self.apply_scaling(train_df)
        valid_df = self.apply_scaling(valid_df)

        os.makedirs(self.output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        valid_df.to_csv(os.path.join(self.output_dir, "valid.csv"), index=False)

        print(f"[DONE] Split saved: train={len(train_df)}, valid={len(valid_df)}")

    def label_encode_categorical(self):
        """Label-encode categorical columns and optionally persist encoders."""
        cat_uniq_val_orders = {}
        for col in self.cat_cols:
            le = LabelEncoder()
            values = self.df[col].astype(str).fillna("__NA__")
            self.df[col] = le.fit_transform(values)
            self.encoders[col] = le
            cat_uniq_val_orders[col] = le.classes_.tolist()
        
        if self.encoders and self.artifacts_dir and self.dump_label_encoders:
            os.makedirs(self.artifacts_dir, exist_ok=True)
            with open(os.path.join(self.artifacts_dir, "label_encoders.pkl"), "wb") as f:
                pickle.dump(self.encoders, f)
            with open(os.path.join(self.artifacts_dir, "cat_uniq_val_orders.json"), "w", encoding="utf-8") as f:
                json.dump(cat_uniq_val_orders, f, ensure_ascii=False, indent=2)
        return self.df

    def numerical_to_sequence(self):
        """Convert numeric columns into a text sequence per row."""
        def row_to_text(row):
            parts = []
            for c in self.num_cols:
                v = row[c]
                s = "NA" if pd.isna(v) else self.float_format.format(v)
                parts.append(f"{c}: {s}")

            if self.shuffle_num_cols_in_text:
                row_id = int(row[self.row_id_col])
                rng = np.random.RandomState(self.random_state + row_id)
                rng.shuffle(parts)

            return " ".join(parts)
        self.df["text"] = self.df.apply(row_to_text, axis=1)

    def preprocess(self, mode="train"):
        """Run label encoding, text serialization, and optional inference scaling."""
        self.label_encode_categorical()
        self.numerical_to_sequence()
        if mode != "train":
            self.apply_scaling(self.df)

        os.makedirs(self.output_dir, exist_ok=True)
        self.df.to_csv(os.path.join(self.output_dir, self.file_name), index=False)
        return 0
