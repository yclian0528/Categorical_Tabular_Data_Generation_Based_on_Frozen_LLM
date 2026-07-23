import os
import json
import torch
from safetensors.torch import load_file
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch.nn.functional as F

from src.utils.config_loader import load_config, set_seed
from src.models.feature_multietask_coshead import FeatureClassifier
from src.data.semantic_categorical_auto_encoder import SemanticCategoricalEncoder

class MMoECategoricalGenerator:
    """Generate categorical columns using a trained feature classifier."""

    def __init__(self, backbone_path, classifier_model_dir, artifacts_dir, cfg):
        self.cfg = cfg
        self.device = cfg.project.device
        
        summary_path = os.path.join(classifier_model_dir, "experiment_summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Experiment summary not found: {summary_path}")
            
        with open(summary_path, "r", encoding="utf-8") as f:
            self.exp_info = json.load(f)
        
        self.task_cols = self.exp_info["task_cols"]
        self.class_dims = self.exp_info["class_dims"]
        self.input_dim = self.exp_info["input_dim"]
        self.proj_dim = self.exp_info["proj_dim"]
        self.raw_num_dim = self.exp_info["raw_num_dim"]
        self.use_mmoe = self.exp_info["use_mmoe"]
        self.num_experts = self.exp_info["num_experts"]
        self.concat_raw_nums = self.exp_info["concat_raw_nums"]
        self.use_attention = self.exp_info.get("use_attention", False)
        self.attention_pos = self.exp_info.get("attention_pos", "after")
        self.attention_heads = self.exp_info.get("attention_heads", None)
        self.use_deep_proj_layers = self.exp_info.get("use_deep_proj_layers", True)

        m_name = backbone_path.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "[PAD]"

        self.torch_dtype = torch.bfloat16 if ("qwen" in m_name or "llama" in m_name) and torch.cuda.is_bf16_supported() else torch.float32
        self.backbone = AutoModel.from_pretrained(
            backbone_path, local_files_only=True, torch_dtype=self.torch_dtype
        ).to(self.device).eval()

        backbone_id = backbone_path.split('/')[-1]
        enc_folder = f"{backbone_id}_{cfg.extraction.prompt_id}_sce"
            
        self.sce = SemanticCategoricalEncoder(model_name_or_path=backbone_path, load_mode=True)
        self.sce.load(os.path.join(artifacts_dir, enc_folder))
        self.label_embs_dict = {c: self.sce.get_embeddings(c).to(self.device) for c in self.task_cols}
        self.label_lists_dict = {c: self.sce.get_label_list(c) for c in self.task_cols}

        self.classifier = FeatureClassifier(
            input_dim=self.input_dim,
            class_dims=self.class_dims,
            raw_num_dim=self.raw_num_dim,
            concat_raw_nums=self.concat_raw_nums,
            use_mmoe=self.use_mmoe,
            num_experts=self.num_experts,
            use_attention=self.use_attention,
            attention_heads=self.attention_heads,
            attention_pos=self.attention_pos,
            proj_dim=self.proj_dim,
            use_deep_proj_layers=self.use_deep_proj_layers
        ).to(self.device).eval()
        
        weight_path = os.path.join(classifier_model_dir, "model.safetensors")
        self.classifier.load_state_dict(load_file(weight_path, device=self.device), strict=False)

    @torch.no_grad()
    def _extract_features(self, texts):
        """Extract frozen-backbone features for a batch of texts."""
        all_feats = []
        ext_batch = self.cfg.extraction.batch_size
        pooling_type = self.sce.pooling_type

        for i in range(0, len(texts), ext_batch):
            batch_texts = texts[i : i + ext_batch]
            enc = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                 max_length=self.cfg.extraction.max_length, return_tensors="pt").to(self.device)
            outputs = self.backbone(**enc)
            last_hidden = outputs.last_hidden_state

            if pooling_type == "cls":
                feat = last_hidden[:, 0, :]
            elif pooling_type == "last":
                sequence_lengths = enc["attention_mask"].sum(dim=1) - 1
                batch_size = last_hidden.shape[0]
                feat = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
            else: 
                mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
                feat = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
            
            is_embedding_model = any(name in self.sce.model_name_or_path.lower() for name in ['bge', 'e5', 'embedding'])
            if is_embedding_model:
                feat = F.normalize(feat, p=2, dim=-1)

            all_feats.append(feat.float().cpu()) 
        return torch.cat(all_feats, dim=0)

    @torch.no_grad()
    def generate(self, numeric_df, temperature=0.8):
        """Generate categorical values for a preprocessed numeric dataframe."""
        texts = numeric_df["text"].tolist()
        inf_batch = self.cfg.generation.batch_size
        all_generated_parts = []
        scaled_cols = [c for c in numeric_df.columns if c.endswith("_scaled")]

        for i in range(0, len(texts), inf_batch):
            batch_texts = texts[i : i + inf_batch]
            
            batch_features = self._extract_features(batch_texts).to(self.device)

            batch_raw_nums = None
            if self.concat_raw_nums and scaled_cols:
                raw_val = numeric_df.iloc[i : i + inf_batch][scaled_cols].values
                batch_raw_nums = torch.tensor(raw_val, dtype=torch.float32).to(self.device)
            
            outputs = self.classifier(
                sequence_repr=batch_features, 
                raw_nums=batch_raw_nums, 
                label_embs_dict=self.label_embs_dict
            )
            logits_dict = outputs["logits"]

            batch_res = {}
            for col in self.task_cols:
                logits = logits_dict[col]
                if temperature <= 1e-6:
                    indices = torch.argmax(logits, dim=-1)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                label_list = self.label_lists_dict[col]
                batch_res[col] = [label_list[idx] for idx in indices.cpu().numpy()]
            
            all_generated_parts.append(pd.DataFrame(batch_res))
            
            del batch_features, outputs

        return pd.concat(all_generated_parts, axis=0).reset_index(drop=True)

def main():
    """Generate complete synthetic tables for all configured experiments."""
    cfg = load_config()
    
    total = len(cfg.generation.temperatures) * len(cfg.experiment_targets.backbones) * \
            len(cfg.experiment_targets.num_gen_models) * len(cfg.experiment_targets.datasets)
    main_pbar = tqdm(total=total, desc="Full Synthesis Pipeline")

    experiment_base = os.path.join(cfg.paths.experiment_root)

    for temp in cfg.generation.temperatures:
        for current_seed in cfg.project.seeds: 
            for bb_name in cfg.experiment_targets.backbones:

                set_seed(current_seed)

                experiment_tag = (
                    f"mmoe_{cfg.model.use_mmoe}_"
                    f"concat_{cfg.model.concat_raw_nums}_"
                    f"sce_{cfg.extraction.prompt_id}_"
                    f"loss_{cfg.training.loss_cfg.type}_"
                    f"deepproj_{cfg.model.use_deep_proj_layers}_"
                    f"attn_{cfg.model.use_attention}"
                )


                if cfg.model.use_attention:
                    experiment_tag += f"_{cfg.model.attention_pos}"

                for num_model in cfg.experiment_targets.num_gen_models:
                    for data_name in cfg.experiment_targets.datasets:
                        
                        numeric_path = os.path.join(
                            cfg.paths.syn_numeric_root,
                            num_model,
                            "syn_data",
                            f"seed_{current_seed}",
                            data_name,
                            f"syn_{data_name}.csv"
                        )

                        classifier_dir = os.path.join(
                            experiment_base,
                            experiment_tag,
                            f"seed_{current_seed}",
                            bb_name,
                            data_name,
                            "final_model"
                        )

                        artifacts_dir = os.path.join(cfg.paths.processed_data_root, data_name, "artifacts")
                        
                        output_dir = os.path.join(
                            cfg.paths.final_syn_output,
                            f"temp_{temp}",
                            experiment_tag,
                            f"seed_{current_seed}",
                            bb_name,
                            num_model,
                            data_name
                        )

                        os.makedirs(output_dir, exist_ok=True)

                        if not os.path.exists(numeric_path) or not os.path.exists(classifier_dir):
                            main_pbar.update(1); continue

                        generator = MMoECategoricalGenerator(
                            backbone_path=os.path.join(cfg.paths.offline_models_dir, bb_name),
                            classifier_model_dir=classifier_dir,
                            artifacts_dir=artifacts_dir,
                            cfg=cfg
                        )

                        df_numeric = pd.read_csv(numeric_path)
                        df_cat = generator.generate(df_numeric, temperature=temp)

                        df_final = pd.concat([df_numeric.reset_index(drop=True), df_cat], axis=1)
                        
                        drop_cols = []
                        if cfg.generation.drop_scaled_cols:
                            drop_cols += [c for c in df_final.columns if c.endswith("_scaled")]
                            if "text" in df_final.columns: drop_cols.append("text")
                        if cfg.generation.drop_row_id:
                            if "__row_id__" in df_final.columns: drop_cols.append("__row_id__")
                        
                        df_final = df_final.drop(columns=drop_cols, errors="ignore")
                        
                        df_final.to_csv(os.path.join(output_dir, f"syn_{data_name}.csv"), index=False)
                        
                        del generator
                        main_pbar.update(1)

    main_pbar.close()

if __name__ == "__main__":
    main()
