import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

def train_model(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate_model(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Classification with Transfer Learning on CIFAR-10")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and validation")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--output_model', type=str, default='transfer_model.pth', help="Path to save the trained model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms for training and validation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Download CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer with a new classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)  # CIFAR-10 has 10 classes

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Only train the final layer parameters
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    best_val_accuracy = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_model(model, criterion, optimizer, train_loader, device)
        val_loss, val_acc = validate_model(model, criterion, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), args.output_model)

    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to {args.output_model}")
