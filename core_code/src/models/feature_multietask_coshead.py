import torch
import torch.nn as nn
from src.utils.config_loader import load_config
from .mmoe_layer import MMoELayer
from .head_utils import NumProjectionHead, CatProjectionHead, CosineHead
from .loss_utils import make_criterion

class FeatureClassifier(nn.Module):
    """Multi-task classifier over frozen backbone features and SCE prototypes."""

    def __init__(
        self,
        input_dim,
        class_dims,
        raw_num_dim=0,
        concat_raw_nums=None,
        use_mmoe=None,
        num_experts=None,
        use_attention=None,
        attention_pos=None,
        attention_heads=None,
        proj_dim=None,
        loss_cfg=None,
        class_weights=None,
        use_deep_proj_layers=None,
        **kwargs
    ):
        super().__init__()
        cfg = load_config()
        
        self.cols = list(class_dims.keys())
        self.use_mmoe = use_mmoe if use_mmoe is not None else cfg.model.use_mmoe
        self.use_attention = use_attention if use_attention is not None else cfg.model.get('use_attention', False)
        self.concat_raw_nums = concat_raw_nums if concat_raw_nums is not None else cfg.model.concat_raw_nums
        self.proj_dim = proj_dim if proj_dim is not None else cfg.model.proj_dim
        self.raw_num_embed_dim = cfg.model.raw_num_embed_dim
        self.attn_pos = attention_pos if attention_pos is not None else cfg.model.get('attention_pos', 'after')
        self.use_deep_proj_layers = use_deep_proj_layers if use_deep_proj_layers is not None else cfg.model.get('use_deep_proj_layers', True)

        self.effective_input_dim = input_dim
        if self.concat_raw_nums and raw_num_dim > 0:
            self.raw_num_projector = nn.Sequential(
                nn.Linear(raw_num_dim, self.raw_num_embed_dim),
                nn.ReLU(),
                nn.LayerNorm(self.raw_num_embed_dim)
            )
            self.effective_input_dim = input_dim + self.raw_num_embed_dim
        
        if self.use_attention and self.attn_pos == "before":
            self.task_embeddings = nn.Parameter(torch.randn(len(self.cols), self.effective_input_dim))
        
        if self.use_mmoe:
            actual_experts = num_experts if num_experts is not None else cfg.model.mmoe.expert_limits[0]
            self.mmoe = MMoELayer(
                input_dim=self.effective_input_dim,
                expert_dim=self.effective_input_dim,
                num_experts=actual_experts,
                num_tasks=len(self.cols)
            )

        if self.use_attention:
            self.attn_heads = attention_heads if attention_heads is not None else cfg.model.get('fixed_attention_heads', 8)
            assert self.effective_input_dim % self.attn_heads == 0, \
                f"Dimension: {self.effective_input_dim} can't be divided by {self.attn_heads} heads."
            
            self.task_attention = nn.MultiheadAttention(
                embed_dim=self.effective_input_dim,
                num_heads=self.attn_heads,
                batch_first=True
            )
            self.attn_norm = nn.LayerNorm(self.effective_input_dim)

        self.num_projs = nn.ModuleDict({
            c: NumProjectionHead(self.effective_input_dim, self.proj_dim, use_deep=self.use_deep_proj_layers) for c in self.cols
        })
        
        self.cat_projs = nn.ModuleDict({
            c: CatProjectionHead(input_dim, self.proj_dim, use_deep=self.use_deep_proj_layers) for c in self.cols
        })
        
        self.cosheads = nn.ModuleDict({
            c: CosineHead(
                hidden_size=self.proj_dim,
                init_scale=cfg.model.cosine_head.init_scale,
                max_scale=cfg.model.cosine_head.max_scale,
                min_scale=cfg.model.cosine_head.min_scale
            ) for c in self.cols
        })
        
        self.loss_config = loss_cfg if loss_cfg is not None else {c: {"type": cfg.training.loss_cfg.type} for c in self.cols}
        self.criteria = nn.ModuleDict()
        for c in self.cols:
            cfg_c = self.loss_config.get(c, {"type": cfg.training.loss_cfg.type})
            self.criteria[c] = make_criterion(cfg_c, class_weight=class_weights.get(c) if class_weights else None)

    def forward(self, sequence_repr, raw_nums=None, labels=None, label_embs_dict=None):
        """Return per-task logits and optional training losses."""
        x_semantic = sequence_repr.float()

        if self.concat_raw_nums and raw_nums is not None:
            x_raw_proj = self.raw_num_projector(raw_nums.float())
            x = torch.cat([x_semantic, x_raw_proj], dim=-1)
        else:
            x = x_semantic
        
        if self.use_attention and self.attn_pos == "before":
            task_feats = x.unsqueeze(1).repeat(1, len(self.cols), 1) 
            task_feats = task_feats + self.task_embeddings.unsqueeze(0)
            
            attn_output, _ = self.task_attention(task_feats, task_feats, task_feats)
            task_feats = self.attn_norm(task_feats + attn_output)
            
            final_task_list = []
            for i, c in enumerate(self.cols):
                feat = self.mmoe(task_feats[:, i, :], i) if self.use_mmoe else task_feats[:, i, :]
                final_task_list.append(feat)
            all_task_feats = torch.stack(final_task_list, dim=1)

        else:
            task_feats_list = []
            for i, c in enumerate(self.cols):
                feat = self.mmoe(x, i) if self.use_mmoe else x
                task_feats_list.append(feat)
            
            all_task_feats = torch.stack(task_feats_list, dim=1)

            if self.use_attention:
                attn_output, _ = self.task_attention(all_task_feats, all_task_feats, all_task_feats)
                all_task_feats = self.attn_norm(all_task_feats + attn_output)

        logits = {}
        total_loss = 0
        per_task_losses = {}
        
        for i, c in enumerate(self.cols):
            task_feat_final = all_task_feats[:, i, :]
            
            q_feat = self.num_projs[c](task_feat_final)
            p_feat = self.cat_projs[c](label_embs_dict[c].to(x.device).float())
            
            logits[c] = self.cosheads[c](q_feat, p_feat)
            
            if labels is not None and c in labels:
                task_loss = self.criteria[c](logits[c], labels[c])
                total_loss += task_loss
                per_task_losses[c] = task_loss.detach()

        return {"loss": total_loss, "logits": logits, "per_task_losses": per_task_losses}
