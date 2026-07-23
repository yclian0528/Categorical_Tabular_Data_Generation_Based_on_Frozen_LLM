import os
import json
import torch
import numpy as np
import math
import shutil

from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.utils.config_loader import load_config, set_seed
from src.data.multi_task_feature_dataset import MultiTaskFeatureDataset
from src.models.feature_multietask_coshead import FeatureClassifier
from src.data.semantic_categorical_auto_encoder import SemanticCategoricalEncoder

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def suggest_attention_heads(dim, num_tasks, cfg):
    """Choose a valid attention-head count for the task representation."""
    if not cfg.model.use_attention:
        return 0
    else:
        if not cfg.model.auto_attention_heads:
            return cfg.model.fixed_attention_heads
        else:
            if num_tasks <= 5:
                preferred = [4, 2, 8]
            elif num_tasks <= 15:
                preferred = [8, 4, 16]
            else:
                preferred = [16, 8, 32]
                
            for h in preferred:
                if dim % h == 0:
                    return h
                    
            for h in [16, 8, 4, 2, 1]:
                if dim % h == 0:
                    return h
    return 1

def suggest_num_experts(num_tasks, num_samples, cfg):
    """Suggest an MMoE expert count within configured limits."""
    min_limit, max_limit = cfg.model.mmoe.expert_limits
    
    task_base = math.ceil(num_tasks / 2) + 1
    data_scale = math.log10(max(num_samples, 100))
    data_modifier = data_scale / 4.0 
    
    suggested = int(task_base * data_modifier)
    return max(min_limit, min(suggested, max_limit))

def build_compute_metrics(task_cols):
    def compute_metrics_fn(eval_pred):
        preds_tuple = eval_pred.predictions
        labels_tuple = eval_pred.label_ids
        metrics = {}
        summary = {"acc": [], "f1_macro": []}

        for i, c in enumerate(task_cols):
            logits = preds_tuple[i]
            labels = labels_tuple[i]
            preds = np.argmax(logits, axis=-1)

            acc = accuracy_score(labels, preds)
            _, _, f_macro, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )

            metrics[f"acc_{c}"] = float(acc)
            metrics[f"f1_macro_{c}"] = float(f_macro)
            summary["acc"].append(acc)
            summary["f1_macro"].append(f_macro)

        if task_cols:
            metrics["acc_mean"] = float(np.mean(summary["acc"]))
            metrics["f1_macro_mean"] = float(np.mean(summary["f1_macro"]))
        return metrics
    return compute_metrics_fn

def feature_collate_fn(batch, task_cols):
    """Stack precomputed features and labels for Trainer batches."""
    seq_reps = torch.stack([item["sequence_repr"] for item in batch]).pin_memory()
    labels = {c: torch.tensor([item[c] for item in batch], dtype=torch.long).pin_memory() for c in task_cols}
    inputs = {"sequence_repr": seq_reps, **labels}
    if "raw_nums" in batch[0]:
        inputs["raw_nums"] = torch.stack([item["raw_nums"] for item in batch]).pin_memory()
    return inputs

class MultiHeadFeatureTrainer(Trainer):
    """Trainer wrapper for dict-based multi-head classifier outputs."""

    def __init__(self, *args, task_cols, label_embs_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_cols = task_cols
        self.label_embs_dict = {
            k: v.to(self.args.device).float() for k, v in (label_embs_dict or {}).items()
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = {c: inputs.pop(c) for c in list(inputs.keys()) if c in self.task_cols}
        outputs = model(**inputs, labels=labels, label_embs_dict=self.label_embs_dict)
        loss = outputs["loss"]

        if self.state.global_step % self.args.logging_steps == 0:
            prefix = "train_" if model.training else "eval_"
            logs = {}
            with torch.no_grad():
                for c in self.task_cols:
                    head = model.cosheads[c]
                    s = torch.nn.functional.softplus(head.raw_scale)
                    logs[f"{prefix}scale_{c}"] = torch.clamp(s, head.min_scale, head.max_scale).detach().cpu().item()
            
            if "per_task_losses" in outputs:
                logs.update({f"{prefix}loss_{k}": v.detach().cpu().item() for k, v in outputs["per_task_losses"].items()})
            self.log(logs)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = {c: inputs.pop(c) for c in list(inputs.keys()) if c in self.task_cols}
        with torch.no_grad():
            outputs = model(**inputs, labels=labels, label_embs_dict=self.label_embs_dict)
        loss = outputs.get("loss", None)
        if prediction_loss_only: return (loss, None, None)
        logits_tuple = tuple(outputs["logits"][c].detach() for c in self.task_cols)
        labels_tuple = tuple(labels[c].detach() for c in self.task_cols) if labels else None
        return (loss, logits_tuple, labels_tuple)

def main():
    """Train a multi-task classifier for each configured seed/backbone/dataset."""
    cfg = load_config()

    data_names = cfg.experiment_targets.datasets
    backbone_names = cfg.experiment_targets.backbones

    for current_seed in cfg.project.seeds:
        print(f"[INFO] Starting training seed: {current_seed}")

        for backbone_name in backbone_names:
            for data_name in data_names:
                set_seed(current_seed)
                print(f"[INFO] Training model: dataset={data_name}, backbone={backbone_name}, seed={current_seed}")

                feature_base = os.path.join(cfg.paths.feature_extract_root, f"seed_{current_seed}", backbone_name, data_name)
                artifacts_dir = os.path.join(cfg.paths.processed_data_root, data_name, "artifacts")
                
                train_ds = MultiTaskFeatureDataset(os.path.join(feature_base, "train_features.pt"))
                valid_ds = MultiTaskFeatureDataset(os.path.join(feature_base, "valid_features.pt"))
                with open(os.path.join(artifacts_dir, "class_dims.json"), "r") as f:
                    class_dims = json.load(f)
                task_cols = list(class_dims.keys())

                input_dim = train_ds.config["hidden_size"]
                raw_num_dim = train_ds.config.get("raw_num_dim", 0)
                
                effective_dim = input_dim
                if cfg.model.concat_raw_nums and raw_num_dim > 0:
                    effective_dim = input_dim + cfg.model.raw_num_embed_dim
                
                semantic_dir = os.path.join(artifacts_dir, f"{backbone_name}_{cfg.extraction.prompt_id}_sce")
                encoder = SemanticCategoricalEncoder(model_name_or_path=os.path.join(cfg.paths.offline_models_dir, backbone_name), load_mode=True)
                encoder.load(semantic_dir)
                label_embs_dict = {c: encoder.get_embeddings(c) for c in task_cols}

                def get_weights(labels_dict, c_dims, rw_cfg):
                    """Compute class-balanced weights from training labels."""
                    weights = {}
                    beta = rw_cfg.beta
                    for col, tensor in labels_dict.items():
                        counts = torch.bincount(tensor, minlength=c_dims[col]).float()
                        eff = (1 - beta**counts.clamp_min(1)) / (1 - beta)
                        w = (1.0 / eff) / ((1.0 / eff).mean() + 1e-12)
                        weights[col] = w.clamp(rw_cfg.lo_clip, rw_cfg.hi_clip)
                    return weights
                
                per_head_weights = get_weights(train_ds.labels, class_dims, cfg.training.loss_reweight)

                num_experts = suggest_num_experts(len(task_cols), len(train_ds), cfg)
                num_attn_heads = suggest_attention_heads(effective_dim, len(task_cols), cfg)
                model = FeatureClassifier(
                    input_dim=train_ds.config["hidden_size"],
                    class_dims=class_dims,
                    use_mmoe=cfg.model.use_mmoe,
                    raw_num_dim=train_ds.config.get("raw_num_dim", 0),
                    concat_raw_nums=cfg.model.concat_raw_nums,
                    num_experts=num_experts,
                    use_attention=cfg.model.use_attention,
                    attention_heads=num_attn_heads,
                    proj_dim=cfg.model.proj_dim,
                    class_weights=per_head_weights, 
                    use_deep_proj_layers=cfg.model.use_deep_proj_layers
                )

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
                
                experiment_root = os.path.join(cfg.paths.experiment_root, experiment_tag, f"seed_{current_seed}", backbone_name, data_name)
                
                final_model_dir = os.path.join(experiment_root, "final_model")
                log_dir = os.path.join(experiment_root, "logs")
                checkpoint_dir = os.path.join(experiment_root, "checkpoints")
                
                for d in [final_model_dir, log_dir, checkpoint_dir]: os.makedirs(d, exist_ok=True)

                training_args = TrainingArguments(
                    output_dir=checkpoint_dir,
                    logging_dir=log_dir,
                    per_device_train_batch_size=cfg.training.trainer.batch_size,
                    learning_rate=cfg.training.optimizer.lr,
                    num_train_epochs=cfg.training.trainer.epochs,
                    lr_scheduler_type=cfg.training.optimizer.lr_scheduler,
                    warmup_ratio=cfg.training.optimizer.warmup_ratio,
                    weight_decay=cfg.training.optimizer.weight_decay,
                    save_total_limit=cfg.training.trainer.save_total_limit,
                    max_grad_norm=cfg.training.trainer.max_grad_norm,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    logging_steps=cfg.training.trainer.logging_steps,
                    load_best_model_at_end=True,
                    metric_for_best_model=cfg.training.trainer.metric_for_best_model,
                    report_to=["tensorboard"],
                    remove_unused_columns=False,
                    dataloader_pin_memory=True
                )

                patience = cfg.training.get("early_stopping", {}).get("patience", 15)

                trainer = MultiHeadFeatureTrainer(
                    model=model, args=training_args, train_dataset=train_ds, eval_dataset=valid_ds,
                    data_collator=lambda b: feature_collate_fn(b, task_cols),
                    task_cols=task_cols, compute_metrics=build_compute_metrics(task_cols),
                    label_embs_dict=label_embs_dict,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
                )

                train_output = trainer.train()
                
                actual_epochs = getattr(trainer.state, "epoch", None)
                if actual_epochs is None:
                    try:
                        actual_epochs = float(train_output.metrics.get("epoch")) if getattr(train_output, "metrics", None) else None
                    except Exception:
                        actual_epochs = None
                if actual_epochs is None:
                    actual_epochs = float(cfg.training.trainer.epochs)

                trainer.save_model(final_model_dir)

                config_src = os.path.join("src", "config", "config.yaml")
                if os.path.exists(config_src):
                    shutil.copy2(config_src, os.path.join(final_model_dir, "exp_config.yaml"))

                exp_info = {
                    "seed": current_seed,
                    "backbone": backbone_name, 
                    "dataset": data_name, 
                    "task_cols": task_cols,
                    "class_dims": class_dims,
                    "input_dim": train_ds.config["hidden_size"],
                    "proj_dim": cfg.model.proj_dim,
                    "raw_num_dim": train_ds.config.get("raw_num_dim", 0),
                    "use_mmoe": cfg.model.use_mmoe,
                    "num_experts": num_experts, 
                    "concat_raw_nums": cfg.model.concat_raw_nums,
                    "use_attention": cfg.model.use_attention,
                    "attention_pos": cfg.model.get('attention_pos', 'after'),
                    "attention_heads": num_attn_heads,
                    "use_deep_proj_layers": cfg.model.get('use_deep_proj_layers', True)
                }
                try:
                    exp_info["actual_epochs"] = float(actual_epochs)
                except Exception:
                    exp_info["actual_epochs"] = actual_epochs
                exp_info["global_step"] = getattr(trainer.state, "global_step", None)
                exp_info["best_model_checkpoint"] = getattr(trainer.state, "best_model_checkpoint", None)

                with open(os.path.join(final_model_dir, "experiment_summary.json"), "w") as f:
                    json.dump(exp_info, f, indent=4)

                print(f"[INFO] Evaluating model: dataset={data_name}, backbone={backbone_name}, seed={current_seed}")
                final_metrics = trainer.evaluate()
                
                results_path = os.path.join(experiment_root, "all_results.json")
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(final_metrics, f, indent=4)
                print(f"[DONE] Results saved: {results_path}")

                del model, trainer

if __name__ == "__main__":
    main()
