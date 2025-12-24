import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from model import BirdClassifier
from dataset import get_dataloader


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

    return running_loss / total, correct / total


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, classes = get_dataloader(
        args.csv_path,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = BirdClassifier(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    mlflow.log_params({
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr
    })

    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        mlflow.log_metrics({
            "loss": loss,
            "accuracy": acc
        }, step=epoch)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss: {loss:.4f} | Acc: {acc:.4f}"
        )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    mlflow.log_artifact(args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="birds_3000.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_path", default="models/model.pt")

    args = parser.parse_args()

    mlflow.start_run()
    main(args)
    mlflow.end_run()
