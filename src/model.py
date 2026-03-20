import torch.nn as nn
from torchvision import models


def build_model(model_name:str, pretrained:bool, num_classes:int = 2,
                dropout:float = 0.5, freeze_backbone:bool = True):
    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        if freeze_backbone:
            _freeze_early_layers(m)
        m.fc = nn.Sequential( 
            nn.Dropout(dropout),
            nn.Linear(m.fc.in_features, num_classes)
        )
        return m
    
    if model_name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        if freeze_backbone:
            _freeze_early_layers(m)
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(m.fc.in_features, num_classes)
        )
        return m

    raise ValueError(f"Unsupported model name: {model_name}")


def _freeze_early_layers(m: nn.Module):
    """
    Freeze all layers except layer4 and fc.
    Only layer4 and the classifier head will be updated during training.
    This reduces overfitting by limiting the number of trainable parameters
    and preserving low-level pretrained features.
    """
    for name, param in m.named_parameters():
        if not name.startswith("layer4") and not name.startswith("fc"):
            param.requires_grad = False
