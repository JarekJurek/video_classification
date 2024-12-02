import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets import FrameVideoDataset
from models.three_d_cnn import ThreeDCNN
from utils import plot_training_metrics


def train(model, optimizer, train_dataloader, val_dataloader, loss_function, num_epochs=10, device="cuda"):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for frames, labels in tqdm(train_dataloader, desc="Training"):
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)  # Pass video batch through model
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
        for frames, labels in tqdm(dataloader, desc="Validating"):
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)  # Pass video batch through model
            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    return val_loss, val_accuracy


def test(model, test_dataloader, device, output_dir="output"):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (frames, labels) in enumerate(tqdm(test_dataloader, desc="Testing")):
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)  # Shape: (B, num_classes)
            _, predicted = torch.max(outputs, 1)  # Shape: (B)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Confusion matrix
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    return accuracy, {"all_labels": all_labels, "all_preds": all_preds}


def main():
    root_dir = "/zhome/a2/c/213547/video_classification/datasets/ufc10"
    output_dir = "output_early_fusion"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreeDCNN(num_classes=10).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = FrameVideoDataset(root_dir=root_dir, split="train", transform=transform, stack_frames=True)
    val_dataset = FrameVideoDataset(root_dir=root_dir, split="val", transform=transform, stack_frames=True)
    test_dataset = FrameVideoDataset(root_dir=root_dir, split="test", transform=transform, stack_frames=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model, train_losses, val_losses, train_accuracies, val_accuracies = train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_function=loss_function,
        num_epochs=10,
        device=device,
    )

    plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_dir=output_dir)

    test(model, test_loader, device, output_dir=output_dir)


if __name__ == "__main__":
    main()
