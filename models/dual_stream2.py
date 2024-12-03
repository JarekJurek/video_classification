import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Spatial Stream using ResNet-18
class SpatialStreamResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(SpatialStreamResNet18, self).__init__()
        # Load the ResNet-18 model pre-trained on ImageNet
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace the last fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# Temporal Stream using ResNet-18
class TemporalStreamResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(TemporalStreamResNet18, self).__init__()
        # Load the ResNet-18 model pre-trained on ImageNet
        self.backbone = models.resnet18(pretrained=True)
        
        # Update the first convolutional layer to handle optical flow input (18 channels)
        self.backbone.conv1 = nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the last fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# Dual Stream Model
class DualStreamModel(nn.Module):
    def __init__(self, spatial_model, temporal_model):
        super(DualStreamModel, self).__init__()
        self.spatial_stream = spatial_model
        self.temporal_stream = temporal_model

    def forward(self, spatial_input, temporal_input):
        """
        Args:
            spatial_input: Tensor of shape [batch_size, 3, H, W]
            temporal_input: Tensor of shape [batch_size, 18, H, W]
        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        spatial_output = self.spatial_stream(spatial_input)  # [batch_size, num_classes]
        temporal_output = self.temporal_stream(temporal_input)  # [batch_size, num_classes]

        # Fusion: Average the class probabilities
        combined_output = (spatial_output + temporal_output) / 2
        return combined_output