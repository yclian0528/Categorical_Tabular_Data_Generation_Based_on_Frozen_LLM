import os
import yaml
import torch
import numpy as np
import random
from box import ConfigBox

def set_seed(seed):
    """Set all random seeds used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_config(config_file_name="config.yaml"):
    """Load a YAML config."""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    potential_path = os.path.join(current_script_dir, "..", "config", config_file_name)
    
    config_path = os.path.abspath(potential_path)

    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), config_file_name)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {potential_path} or {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            raw_config = yaml.safe_load(f)
            cfg = ConfigBox(raw_config)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse YAML config: {e}")

    if "project" in cfg and "device" in cfg.project:
        if cfg.project.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA is not available. Using CPU.")
            cfg.project.device = "cpu"

    return cfg
