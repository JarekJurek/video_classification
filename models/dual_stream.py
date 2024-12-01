from torchvision.models import EfficientNet_V2_S_Weights
import torch.nn as nn
from torchvision import models

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
    
spatial_model = PerFrameTrained(num_classes=10)

class TemporalStreamEarlyFusion(nn.Module):
    def __init__(self, num_classes=10):
        super(TemporalStreamEarlyFusion, self).__init__()
        # Early fusion: First Conv layer processes all temporal frames together
        self.conv1 = nn.Conv2d(in_channels=20, out_channels=64, kernel_size=7, stride=2, padding=3)  # 2*(T-1) = 20 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 7 * 7, 2048)  # Assuming input size 224x224
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        return x

# Temporal model initialization
temporal_model = TemporalStreamEarlyFusion(num_classes=10)

class DualStreamModel(nn.Module):
    def __init__(self, spatial_model, temporal_model):
        super(DualStreamModel, self).__init__()
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model

    def forward(self, spatial_input, temporal_input):
        spatial_output = self.spatial_model(spatial_input)  # Spatial stream predictions
        temporal_output = self.temporal_model(temporal_input)  # Temporal stream predictions
        
        # Fusion: Average the class probabilities
        combined_output = (spatial_output + temporal_output) / 2
        return combined_output

# Initialize the dual-stream model
dual_stream_model = DualStreamModel(spatial_model, temporal_model)


