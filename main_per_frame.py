import torch
import os
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from models.PerFrameCNN import PerFrameTrained
from datasets import FrameImageDataset, FrameVideoDataset
from tqdm import tqdm
from utils import plot_training_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
            outputs = model(frames)
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
            outputs = model(frames)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    return val_loss, val_accuracy


def test(model, test_dataloader, device, output_dir="output"):
    model.eval()
    results = {}  # To store video-level predictions
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for frames, labels in test_dataloader:
            frames = frames.to(device)
            labels = labels.to(device)

            # Flatten frames for frame-level predictions
            batch_size, channels, num_frames, height, width = frames.size()
            frames = frames.permute(0, 2, 1, 3, 4)  # [batch_size, num_frames, channels, height, width]
            frames = frames.reshape(-1, channels, height, width)  # [batch_size * num_frames, channels, height, width]

            outputs = model(frames)  # [batch_size * num_frames, num_classes]

            # Reshape outputs back to video-level
            outputs = outputs.reshape(batch_size, num_frames, -1)  # [batch_size, num_frames, num_classes]

            aggregated_outputs = torch.mean(outputs, dim=1)  # [batch_size, num_classes]

            _, predicted = torch.max(aggregated_outputs, 1)  # [batch_size]

            # Store results and compute accuracy
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(batch_size):
                results[i] = predicted[i].item()
                correct += (predicted[i] == labels[i]).item()
                total += 1

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    return results



def main():
    root_dir = "/zhome/a2/c/213547/video_classification/datasets/ufc10"
    output_dir = "output_per_frame"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerFrameTrained(num_classes=10).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = FrameImageDataset(root_dir=root_dir, split="train", transform=transform)
    val_dataset = FrameImageDataset(root_dir=root_dir, split="val", transform=transform)
    test_dataset = FrameVideoDataset(root_dir=root_dir, split="test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

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
