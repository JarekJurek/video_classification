import torch
import torch.nn as nn
from torchvision.models.video import r3d_18  # ResNet3D-18

class ResNet3DClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ResNet3DClassifier, self).__init__()
        # Load ResNet3D-18 backbone
        self.backbone = r3d_18(pretrained=pretrained)
        
        # Replace the classifier head
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [Batch, Channels, NumFrames, Height, Width]
        Returns:
            video_logits: Tensor of shape [Batch, NumClasses]
        """
        return self.backbone(x)
