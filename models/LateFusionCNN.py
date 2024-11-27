import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s

class LateFusionEfficientNetV2Small(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(LateFusionEfficientNetV2Small, self).__init__()
        # Load pre-trained EfficientNet V2 Small
        self.backbone = efficientnet_v2_s(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove classifier
        
        # Fully connected layer for feature aggregation
        self.fc_fusion = nn.Sequential(
            nn.Linear(self.backbone.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames):
        """
        Args:
            frames: Tensor of shape [Batch, Channels, NumFrames, Height, Width]
        Returns:
            video_logits: Tensor of shape [Batch, NumClasses]
        """
        batch_size, channels, num_frames, height, width = frames.size()
        frames = frames.permute(0, 2, 1, 3, 4)  # Reshape to [Batch, NumFrames, Channels, Height, Width]

        # Process each frame independently through the backbone
        frame_features = []
        for i in range(num_frames):
            frame = frames[:, i, :, :, :]  # [Batch, Channels, Height, Width]
            feature = self.feature_extractor(frame)  # Extract spatial features
            feature = torch.flatten(feature, 1)  # Flatten for FC layers
            frame_features.append(feature)

        # Stack frame features and fuse
        frame_features = torch.stack(frame_features, dim=1)  # [Batch, NumFrames, Features]
        fused_features = torch.mean(frame_features, dim=1)  # Mean fusion

        # Final classification
        video_logits = self.fc_fusion(fused_features)
        return video_logits
