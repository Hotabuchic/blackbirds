import torch
import torch.nn as nn
import torch.optim as optim

from model import BirdClassifier
from dataset import get_dataloader

CSV_PATH = "birds_3000.csv"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    print(f"Using device: {DEVICE}")

    train_loader, classes = get_dataloader(
        CSV_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print(f"Classes: {classes}")
    model = BirdClassifier(num_classes=len(classes))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        loss, acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {loss:.4f} | Accuracy: {acc:.4f}"
        )

    torch.save(model.state_dict(), "models/model.pt")


if __name__ == "__main__":
    main()
