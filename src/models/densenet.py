import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from .se import SEBlock

class DenseNet121GAP(nn.Module):
    def __init__(self, num_classes=3, dropout=0.2, pretrained=True, attention="none"):
        super().__init__()
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = m.features  # 1024 channels
        self.relu = nn.ReLU(inplace=True)
        self.attn = SEBlock(1024, r=16) if attention == "se" else nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.attn(x)
        x = self.gap(x)
        return self.classifier(x)
