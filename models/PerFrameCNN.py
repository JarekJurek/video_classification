import torch.nn as nn
from torchvision import models


class PerFrameOwn(nn.Module):
    def __init__(self, num_classes=10):
        super(PerFrameOwn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PerFrameTrained(nn.Module):
    def __init__(self, num_classes=10):
        super(PerFrameTrained, self).__init__()
        # Load the EfficientNet model
        self.backbone = models.efficientnet_v2_s(pretrained=True)
        
        # Replace the classifier with a custom one
        in_features = self.backbone.classifier[1].in_features  # Get the input features of the classifier
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)  # Replace it with a new linear layer

    def forward(self, x):
        return self.backbone(x)
