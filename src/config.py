from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SUBSET_DIR = DATA_DIR / "raw_subset"
MANIFEST_DIR = DATA_DIR
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
PRED_DIR = RESULTS_DIR / "predictions"

for _d in (DATA_DIR, SUBSET_DIR, RESULTS_DIR, PLOTS_DIR, METRICS_DIR, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

HF_PRIMARY = "aharley/rvl_cdip"
HF_FALLBACKS = ("HuggingFaceM4/rvl_cdip", "jordyvl/rvl_cdip_easyocr")
NUM_CLASSES = 16

N_PER_CLASS_TRAIN = int(os.environ.get("N_TRAIN", 2000))
N_PER_CLASS_VAL = int(os.environ.get("N_VAL", 250))
N_PER_CLASS_TEST = int(os.environ.get("N_TEST", 500))

IMG_SIZE = 224
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 12))
LR = float(os.environ.get("LR", 1e-4))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 8))
USE_AMP = True

DEVICE_INDEX = int(os.environ.get("DEVICE_INDEX", 1))

def get_device():
    import torch

    if torch.cuda.is_available():
        idx = DEVICE_INDEX if DEVICE_INDEX < torch.cuda.device_count() else 0
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")

@dataclass
class ModelSpec:
    key: str
    family: str
    backend: str
    pretrain: str
    hf_or_timm_id: str = ""
    notes: str = ""

MODELS: dict[str, ModelSpec] = {
    "baseline_cnn": ModelSpec("baseline_cnn", "cnn", "scratch", "none",
                              notes="3 blocos conv, piso"),
    "resnet18":     ModelSpec("resnet18", "cnn", "timm", "imagenet",
                              "resnet18"),
    "deit_tiny":    ModelSpec("deit_tiny", "transformer", "timm", "imagenet",
                              "deit_tiny_patch16_224"),
    "mobilevit_s":  ModelSpec("mobilevit_s", "hybrid", "timm", "imagenet",
                              "mobilevit_s"),
    "dit_base":     ModelSpec("dit_base", "transformer", "hf", "domain",
                              "microsoft/dit-base",
                              notes="SSL em 42M docs (IIT-CDIP)"),
    "dit_rvlcdip":  ModelSpec("dit_rvlcdip", "transformer", "hf", "domain+rvlcdip",
                              "microsoft/dit-base-finetuned-rvlcdip",
                              notes="teto de referência (sanity)"),
}
