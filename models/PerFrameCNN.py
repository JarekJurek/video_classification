import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights


class PerFrameTrained(nn.Module):
    def __init__(self, num_classes=10):
        super(PerFrameTrained, self).__init__()
        # Load the EfficientNet model with weights
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1  # Use pre-trained weights
        self.backbone = models.efficientnet_v2_s(weights=weights)
        
        # Replace the classifier with a custom one
        in_features = self.backbone.classifier[1].in_features  # Get the input features of the classifier
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)  # Replace it with a new linear layer

    def forward(self, x):
        return self.backbone(x)
