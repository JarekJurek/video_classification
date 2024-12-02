# models/dual_stream.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights

class PerFrameTrained(nn.Module):
    def __init__(self, num_classes=10):
        super(PerFrameTrained, self).__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1  # Use pre-trained weights
        self.backbone = models.efficientnet_v2_s(weights=weights)
        
        # Replace the classifier with a custom one
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
        
class TemporalStreamEarlyFusion(nn.Module):
    def __init__(self, num_classes=10):
        super(TemporalStreamEarlyFusion, self).__init__()
        # Updated in_channels from 20 to 18
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Update fc1 to match the new feature size
        self.fc1 = nn.Linear(256 * 4 * 4, 2048)  # Changed from 256*7*7 to 256*4*4
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))        # [batch_size, 64, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [batch_size, 128, 16, 16]
        x = self.pool(F.relu(self.conv3(x)))  # [batch_size, 256, 8, 8]
        x = x.view(x.size(0), -1)        # [batch_size, 256*8*8] = [batch_size, 2048]
        x = F.relu(self.fc1(x))          # [batch_size, 2048]
        x = self.fc2(x)                   # [batch_size, num_classes]
        return x

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
