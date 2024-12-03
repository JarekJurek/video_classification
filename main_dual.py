# main.py

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import FrameVideoDataset, FlowVideoDataset  # Use FrameVideoDataset
from models.dual_stream import DualStreamModel, PerFrameTrained, TemporalStreamEarlyFusion
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import plot_training_metrics
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam

class CombinedDataset(Dataset):
    def __init__(self, frame_video_dataset, flow_video_dataset):
        assert len(frame_video_dataset) == len(flow_video_dataset), "Datasets must have the same length"
        self.frame_video_dataset = frame_video_dataset
        self.flow_video_dataset = flow_video_dataset

    def __len__(self):
        return len(self.frame_video_dataset)

    def __getitem__(self, idx):
        frames, label1 = self.frame_video_dataset[idx]
        flows, label2 = self.flow_video_dataset[idx]
        assert label1 == label2, "Labels do not match between datasets"
        return frames, flows, label1

def train(model, optimizer, train_dataloader, val_dataloader, loss_function, num_epochs=10, device="cuda"):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for frames, flows, labels in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            frames = frames.to(device)          # [batch_size, 3, num_frames, 224, 224]
            flows = flows.to(device)            # [batch_size, 18, 64, 64]
            labels = labels.to(device)          # [batch_size]

            optimizer.zero_grad()

            # Forward pass for dual-stream model
            outputs = model(frames, flows)       # [batch_size, num_classes]

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        train_losses.append(train_loss / len(train_dataloader))
        train_accuracies.append(train_accuracy)

        # Validation step
        val_loss, val_accuracy = validate(model, val_dataloader, loss_function, device)
        val_losses.append(val_loss / len(val_dataloader))
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"[TRAIN] Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2%}")
        print(f"[VALIDATION] Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.2%}")
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies

def validate(model, dataloader, loss_function, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, flows, labels in tqdm(dataloader, desc="Validating"):
            frames = frames.to(device)
            flows = flows.to(device)
            labels = labels.to(device)
            
            outputs = model(frames, flows)  # [batch_size, num_classes]

            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    return val_loss, val_accuracy

def test(model, test_loader, device, output_dir):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (frames, flows, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            frames = frames.to(device)
            flows = flows.to(device)
            labels = labels.to(device)

            outputs = model(frames, flows)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2%}")

    # Confusion matrix
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Optionally, save test results
    with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_accuracy:.2%}\n")

def main():
    root_dir = "/dtu/datasets1/02516/ucf101_noleakage"  # Update to your dataset path
    output_dir = "output_dual"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spatial_model = PerFrameTrained(num_classes=10)
    temporal_model = TemporalStreamEarlyFusion(num_classes=10)

    model = DualStreamModel(spatial_model, temporal_model).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_frame_dataset = FrameVideoDataset(root_dir=root_dir, split="train", transform=transform, stack_frames=True)
    train_flow_dataset = FlowVideoDataset(root_dir=root_dir, split="train", resize=(64, 64))

    val_frame_dataset = FrameVideoDataset(root_dir=root_dir, split="val", transform=transform, stack_frames=True)
    val_flow_dataset = FlowVideoDataset(root_dir=root_dir, split="val", resize=(64, 64))

    test_frame_dataset = FrameVideoDataset(root_dir=root_dir, split="test", transform=transform, stack_frames=True)
    test_flow_dataset = FlowVideoDataset(root_dir=root_dir, split="test", resize=(64, 64))

    train_dataset = CombinedDataset(train_frame_dataset, train_flow_dataset)
    val_dataset = CombinedDataset(val_frame_dataset, val_flow_dataset)
    test_dataset = CombinedDataset(test_frame_dataset, test_flow_dataset)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model, train_losses, val_losses, train_accuracies, val_accuracies = train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_function=loss_function,
        num_epochs=15,
        device=device,
    )

    plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_dir=output_dir)

    test(model, test_loader, device, output_dir=output_dir)

if __name__ == "__main__":
    main()
