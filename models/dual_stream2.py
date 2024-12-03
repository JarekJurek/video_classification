import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpatialStreamResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(SpatialStreamResNet18, self).__init__()
        # Load the ResNet-18 model pre-trained on ImageNet
        self.backbone = models.resnet18(pretrained=True)

        # Replace the last fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class TemporalStreamResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(TemporalStreamResNet18, self).__init__()
        # Load ResNet-18 pre-trained on ImageNet
        self.backbone = models.resnet18(pretrained=True)

        # Modify the first convolutional layer to accept 18-channel input
        self.backbone.conv1 = nn.Conv2d(
            in_channels=18,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace the last fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class SpatialStream(nn.Module):
    def __init__(self, spatial_model):
        super(SpatialStream, self).__init__()
        self.spatial_model = spatial_model

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, 3, num_frames, H, W]
        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        batch_size, channels, num_frames, height, width = x.size()
        # Permute to [batch_size, num_frames, channels, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        # Reshape to [batch_size * num_frames, channels, H, W]
        x = x.reshape(batch_size * num_frames, channels, height, width)
        # Pass through the spatial model
        x = self.spatial_model(x)  # [batch_size * num_frames, num_classes]
        # Reshape back to [batch_size, num_frames, num_classes]
        x = x.reshape(batch_size, num_frames, -1)
        # Aggregate by averaging over frames
        x = x.mean(dim=1)  # [batch_size, num_classes]
        return x


class DualStreamModel(nn.Module):
    def __init__(self, spatial_model, temporal_model):
        super(DualStreamModel, self).__init__()
        self.spatial_stream = SpatialStream(spatial_model)
        self.temporal_model = temporal_model

    def forward(self, spatial_input, temporal_input):
        """
        Args:
            spatial_input: Tensor of shape [batch_size, 3, num_frames, H, W]
            temporal_input: Tensor of shape [batch_size, 18, H, W]
        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        spatial_output = self.spatial_stream(spatial_input)  # [batch_size, num_classes]
        temporal_output = self.temporal_model(temporal_input)  # [batch_size, num_classes]

        # Fusion: Average the class probabilities
        combined_output = (spatial_output + temporal_output) / 2
        return combined_output
