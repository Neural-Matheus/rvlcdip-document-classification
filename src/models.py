from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C

class BaselineCNN(nn.Module):

    def __init__(self, num_classes: int = C.NUM_CLASSES, in_ch: int = 3):
        super().__init__()

        def block(ci, co):

            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=False),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, 32), block(32, 64), block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

class HFImageClassifier(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        return self.model(pixel_values=x).logits

    @property
    def config(self):
        return self.model.config

def build_model(spec, num_classes: int = C.NUM_CLASSES,
                grad_checkpointing: bool = False, freeze_backbone: bool = False):

    if spec.backend == "scratch":
        model = BaselineCNN(num_classes)

    elif spec.backend == "timm":
        import timm

        model = timm.create_model(spec.hf_or_timm_id, pretrained=True,
                                  num_classes=num_classes)
        if grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing(True)
        if freeze_backbone:
            _freeze_timm_backbone(model)

    elif spec.backend == "hf":
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(
            spec.hf_or_timm_id,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        if grad_checkpointing:
            model.gradient_checkpointing_enable()
        if freeze_backbone:
            _freeze_hf_backbone(model)
        model = HFImageClassifier(model)

    else:
        raise ValueError(f"backend desconhecido: {spec.backend}")

    return model

def _freeze_timm_backbone(model):
    classifier = model.get_classifier()
    for p in model.parameters():
        p.requires_grad = False
    for p in classifier.parameters():
        p.requires_grad = True

def _freeze_hf_backbone(model):
    for name, p in model.named_parameters():
        p.requires_grad = "classifier" in name

def count_params(model, trainable_only=False) -> int:
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))
