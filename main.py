import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from models.PerFrameCNN import PerFrameTrained
from dataset import FrameImageDataset



def train(model, optimizer, train_dataloader, val_dataloader, loss_function, num_epochs=10, device="cuda"):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for frames, labels in train_dataloader:
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy = validate(model, val_dataloader, loss_function, device)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"[TRAIN] Loss: {train_loss/len(train_dataloader):.4f}, Accuracy: ")
        print(f"[Validation] Loss: {val_loss/len(val_dataloader):.4f}, Accuracy: {val_accuracy}")
    return model


def validate(model, dataloader, loss_function, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in dataloader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    return val_loss, val_accuracy


def main():
    root_dir = "/work3/ppar/data/ucf101"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerFrameTrained(num_classes=10).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = FrameImageDataset(root_dir=root_dir, split="train", transform=transform)
    val_dataset = FrameImageDataset(root_dir=root_dir, split="val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_function=loss_function,
        num_epochs=10,
        device=device,
    )

if __name__ == "__main__":
    main()
