import torch.nn as nn
from torchvision import models

class BirdClassifier(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
