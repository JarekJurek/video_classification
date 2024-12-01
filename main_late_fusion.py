import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from models.LateFusionCNN import LateFusionModel
from datasets import FrameVideoDataset
from tqdm import tqdm


def train(model, optimizer, train_dataloader, val_dataloader, loss_function, num_epochs=10, device="cuda"):
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

        # Validation step
        val_loss, val_accuracy = validate(model, val_dataloader, loss_function, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"[TRAIN] Loss: {train_loss/len(train_dataloader):.4f}, Accuracy: {train_accuracy:.2%}")
        print(f"[VALIDATION] Loss: {val_loss/len(val_dataloader):.4f}, Accuracy: {val_accuracy:.2%}")
    return model



def validate(model, dataloader, loss_function, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Validating"):
            # frames: (B, T, C, H, W), labels: (B)
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)  # Pass video batch through model
            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    return val_loss, val_accuracy



def test(model, test_dataloader, device):
    model.eval()
    correct = 0
    total = 0
    results = {}  # Dictionary to store predictions for each video

    with torch.no_grad():
        for batch_idx, (frames, labels) in enumerate(tqdm(test_dataloader, desc="Testing")):
            # frames: (B, T, C, H, W), labels: (B)
            frames, labels = frames.to(device), labels.to(device)

            outputs = model(frames)  # Shape: (B, num_classes)

            _, predicted = torch.max(outputs, 1)  # Shape: (B)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for i in range(frames.size(0)):  # Iterate through the batch
                results[batch_idx * test_dataloader.batch_size + i] = {
                    "true_label": labels[i].item(),
                    "predicted_label": predicted[i].item(),
                }

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy, results



def main():
    root_dir = "/zhome/a2/c/213547/video_classification/datasets/ufc10"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LateFusionModel(num_classes=10).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = FrameVideoDataset(root_dir=root_dir, split="train", transform=transform)
    val_dataset = FrameVideoDataset(root_dir=root_dir, split="val", transform=transform)
    test_dataset = FrameVideoDataset(root_dir=root_dir, split="test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_function=loss_function,
        num_epochs=3,
        device=device,
    )

    test(model, test_loader, device)


if __name__ == "__main__":
    main()
