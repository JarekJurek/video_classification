import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Spatial Stream Model
class SpatialStreamConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SpatialStreamConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3)  # Input channels = 3
        self.norm1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2)
        self.norm2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, 3, num_frames, H, W]
        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        batch_size, channels, num_frames, height, width = x.size()
        # Reshape: [batch_size, 3, num_frames, H, W] -> [batch_size * num_frames, 3, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

        # Pass through the convolutional layers
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # [batch_size * num_frames, num_classes]

        # Reshape back: [batch_size * num_frames, num_classes] -> [batch_size, num_frames, num_classes]
        x = x.view(batch_size, num_frames, -1)

        # Aggregate over frames
        x = x.mean(dim=1)  # Average over frames -> [batch_size, num_classes]
        return x


# Temporal Stream Model (No Changes Required)
class TemporalStreamEarlyFusion(nn.Module):
    def __init__(self, num_classes=10):
        super(TemporalStreamEarlyFusion, self).__init__()
        self.conv1 = nn.Conv2d(18, 96, kernel_size=7, stride=2, padding=3)  # Input channels = 18
        self.norm1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2)
        self.norm2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 4 * 4, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# Dual Stream Model (No Changes Required)
class DualStreamModel(nn.Module):
    def __init__(self, spatial_model, temporal_model):
        super(DualStreamModel, self).__init__()
        self.spatial_stream = spatial_model
        self.temporal_stream = temporal_model

    def forward(self, spatial_input, temporal_input):
        """
        Args:
            spatial_input: Tensor of shape [batch_size, 3, num_frames, H, W]
            temporal_input: Tensor of shape [batch_size, 18, H, W]
        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        spatial_output = self.spatial_stream(spatial_input)  # [batch_size, num_classes]
        temporal_output = self.temporal_stream(temporal_input)  # [batch_size, num_classes]

        # Fusion: Average the class probabilities
        combined_output = (spatial_output + temporal_output) / 2
        return combined_output