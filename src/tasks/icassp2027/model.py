"""ResNet-50 source classifier for the ICASSP-2027 study.

Architecture (documented, not silently changeable):
    * torchvision ``resnet50``; ``weights`` from the config (``IMAGENET1K_V2`` or ``none``).
    * First convolution is the stock 3-channel conv1 — the input is the grayscale frame replicated
      to three channels (``L -> [L, L, L]``), so no conv1 surgery is performed.
    * ``fc`` replaced by ``Identity`` -> 2048-d global-average-pooled feature (layer ``avgpool``).
    * Optional projection ``Linear(2048, proj_dim) -> BatchNorm1d -> ReLU`` (``proj_dim: 0`` = identity);
      the *feature* used by every analysis is the projection output (or the 2048-d pooled vector).
    * Classifier head: ``Linear(feat_dim, 20)``.
    * Normalization: ImageNet mean/std on the replicated channels (see ``data.INPUT_POLICY``).
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm


class SourceClassifier(nn.Module):
    def __init__(self, num_classes: int = 20, pretrained: str | None = "IMAGENET1K_V2",
                 proj_dim: int = 0):
        super().__init__()
        weights = None
        if pretrained and pretrained.lower() != "none":
            weights = getattr(tvm.ResNet50_Weights, pretrained)
        self.backbone = tvm.resnet50(weights=weights)
        self.backbone.fc = nn.Identity()
        self.pooled_dim = 2048
        if proj_dim and proj_dim > 0:
            self.proj = nn.Sequential(nn.Linear(self.pooled_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True))
            self.feat_dim = proj_dim
        else:
            self.proj = nn.Identity()
            self.feat_dim = self.pooled_dim
        self.head = nn.Linear(self.feat_dim, num_classes)
        self.pretrained = pretrained if weights is not None else "none"
        self.proj_dim = proj_dim or 0
        self.num_classes = num_classes

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f = self.features(x)
        logits = self.head(f)
        return (logits, f) if return_features else logits

    def describe(self) -> dict:
        return {
            "backbone": "torchvision resnet50",
            "pretrained_weights": self.pretrained,
            "first_conv": "stock 3-channel conv1; grayscale input replicated L -> [L, L, L]; no re-initialisation",
            "feature_layer": "avgpool (2048-d) -> projection" if self.proj_dim else "avgpool (2048-d), projection = identity",
            "projection": f"Linear(2048,{self.proj_dim})+BN+ReLU" if self.proj_dim else "identity",
            "feature_dim": self.feat_dim,
            "classifier_head": f"Linear({self.feat_dim}, {self.num_classes})",
            "normalization": "ImageNet mean/std on three replicated channels",
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }
