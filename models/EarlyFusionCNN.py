import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class EarlyFusionEfficientNetV2Small(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(EarlyFusionEfficientNetV2Small, self).__init__()
        
        # Pretrained EfficientNet backbone
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1  # Use pre-trained weights
        self.backbone = efficientnet_v2_s(weights=weights)
        
        # Adjust input layer to handle temporal frames
        in_channels = self.backbone.features[0][0].in_channels
        self.backbone.features[0][0] = nn.Conv3d(
            in_channels=in_channels,  # Typically 3
            out_channels=self.backbone.features[0][0].out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1)
        )
        
        # Global pooling layer
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [Batch, Channels, NumFrames, Height, Width]
        Returns:
            video_logits: Tensor of shape [Batch, NumClasses]
        """
        # Pass through modified EfficientNet backbone
        x = self.backbone.features(x)  # 3D convolutions for spatial-temporal features
        x = self.pool(x)  # Global pooling
        x = self.classifier(x)  # Fully connected classification
        return x
