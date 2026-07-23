import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicProjection(nn.Module):
    """Project high-dimensional inputs into the shared cosine space."""

    def __init__(self, input_dim, proj_dim, use_deep, dropout=0.1):
        super().__init__()
        
        if not use_deep or input_dim <= proj_dim * 2:
            self.proj = nn.Linear(input_dim, proj_dim)
        else:
            layers = []
            curr_dim = input_dim
            while curr_dim > proj_dim * 2:
                next_dim = curr_dim // 2
                layers.append(nn.Linear(curr_dim, next_dim))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(next_dim))
                layers.append(nn.Dropout(dropout))
                curr_dim = next_dim
            
            layers.append(nn.Linear(curr_dim, proj_dim))
            self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x)

class NumProjectionHead(nn.Module):
    """Project sample features into the cosine space."""

    def __init__(self, input_dim, proj_dim, use_deep):
        super().__init__()
        self.proj = DynamicProjection(input_dim, proj_dim, use_deep=use_deep)

    def forward(self, x):
        return self.proj(x)

class CatProjectionHead(nn.Module):
    """Project class-prototype embeddings into the cosine space."""

    def __init__(self, cat_hidden_size, proj_dim, use_deep):
        super().__init__()
        self.proj = DynamicProjection(cat_hidden_size, proj_dim, use_deep=use_deep)

    def forward(self, x):
        return self.proj(x)

class CosineHead(nn.Module):
    """Compute scaled cosine-similarity logits."""

    def __init__(
        self, 
        hidden_size, 
        init_scale=2.0, 
        max_scale=50.0, 
        min_scale=0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_scale = max_scale
        self.min_scale = min_scale

        init_scale_t = torch.tensor(float(init_scale), dtype=torch.float32)
        init_raw = torch.log(torch.exp(init_scale_t) - 1.0 + 1e-6)
        self.raw_scale = nn.Parameter(init_raw)

    def forward(self, query_repr, class_prototypes):
        """Return logits with shape [batch_size, num_classes]."""
        query_repr_norm = F.normalize(query_repr, p=2, dim=-1)
        class_prototypes_norm = F.normalize(class_prototypes, p=2, dim=-1)

        cos_sim = torch.matmul(query_repr_norm, class_prototypes_norm.t())

        scale = F.softplus(self.raw_scale)
        scale = torch.clamp(scale, self.min_scale, self.max_scale)

        logits = scale * cos_sim
        return logits
