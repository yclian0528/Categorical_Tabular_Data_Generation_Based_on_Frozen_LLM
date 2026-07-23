import torch
from torch.utils.data import Dataset

class MultiTaskFeatureDataset(Dataset):
    """Load precomputed backbone features, labels, and optional numeric tensors."""

    def __init__(self, pt_path):
        data = torch.load(pt_path, map_location="cpu")
        
        self.features = data["features"]
        self.labels = data["labels"]
        self.config = data["config"]

        self.raw_nums = data.get("raw_nums", None)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """Return one feature sample with labels for all tasks."""
        item = {c: v[idx] for c, v in self.labels.items()}

        item["sequence_repr"] = self.features[idx].float()

        if self.raw_nums is not None:
            item["raw_nums"] = self.raw_nums[idx].float()
            
        return item
