import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class LateFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(LateFusionModel, self).__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.cnn = efficientnet_v2_s(weights=weights)
        feature_dim = self.cnn.classifier[1].in_features  # Save the feature dimension before removing the classifier
        self.cnn.classifier = nn.Identity()  # Remove the classification head
        self.fc = nn.Linear(feature_dim, num_classes)  # Linear layer for classification

    def forward(self, video_frames):
        """
        Args:
            video_frames: Tensor of shape (B, C, T, H, W)
        Returns:
            output: Tensor of shape (B, num_classes)
        """
        # Permute dimensions to (B, T, C, H, W)
        video_frames = video_frames.permute(0, 2, 1, 3, 4)
        
        batch_size, num_frames, channels, height, width = video_frames.size()
        
        # Reshape to (B*T, C, H, W) to process each frame independently
        video_frames = video_frames.reshape(-1, channels, height, width)
        
        # Extract features for each frame
        frame_features = self.cnn(video_frames)
        
        # Reshape back to (B, T, D)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        # Pool features across time (average)
        pooled_features = torch.mean(frame_features, dim=1)
        
        # Classification
        output = self.fc(pooled_features)
        return output

