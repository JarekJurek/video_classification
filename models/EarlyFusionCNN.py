import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, num_frames=10):
        super(EarlyFusionModel, self).__init__()
        
        self.num_frames = num_frames

        # Pretrained EfficientNet backbone
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1  # Use pre-trained weights
        self.backbone = efficientnet_v2_s(weights=weights)
        
        # Adjust input layer to handle reshaped temporal frames as channels
        in_channels = self.backbone.features[0][0].in_channels  # Typically 3
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=self.num_frames * in_channels,  # T * 3 (e.g., 10 frames * 3 channels)
            out_channels=self.backbone.features[0][0].out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # Global pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
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
            x: Tensor of shape [Batch, Channels, TemporalFrames, Height, Width]
        Returns:
            video_logits: Tensor of shape [Batch, NumClasses]
        """
        # Reshape to [Batch, T*3, H, W] for early fusion
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(B, T * C, H, W)  # [B, T*C, H, W]

        # Pass through modified EfficientNet backbone
        x = self.backbone.features(x)  # 2D convolutions for spatial-temporal fusion
        x = self.pool(x)  # Global pooling
        x = self.classifier(x)  # Fully connected classification
        return x
