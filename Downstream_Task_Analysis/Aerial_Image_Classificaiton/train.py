import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import get_model


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)

g = torch.Generator()
g.manual_seed(42)


# =========================
# Settings
# =========================
train_dir = ""
test_dir = ""
checkpoint_dir = "checkpoints"

model_name = "resnet18"          # resnet18 / resnet50 / efficientnet_b0
batch_size = 256
num_epochs = 50
learning_rate = 1e-4
image_size = 128

os.makedirs(checkpoint_dir, exist_ok=True)
ckpt_path = os.path.join(checkpoint_dir, f"{model_name}.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])


# =========================
# Dataset and DataLoader
# =========================
train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g
)

num_classes = len(train_dataset.classes)

print("Classes:", train_dataset.classes)
print("Number of classes:", num_classes)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))


# =========================
# Model
# =========================
model = get_model(model_name, num_classes)
model = model.to(device)

print(f"Loaded model: {model_name}")
print(f"Checkpoint path: {ckpt_path}")


# =========================
# Loss and Optimizer
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# =========================
# Training Function
# =========================
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels).item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                test_correct += torch.sum(preds == labels).item()
                test_total += labels.size(0)

        test_loss /= test_total
        test_acc = test_correct / test_total

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_weights = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), ckpt_path)
            print(f"Best model saved to: {ckpt_path}")

    model.load_state_dict(best_model_weights)
    print(f"\nBest Test Accuracy: {best_acc:.4f}")

    return model


# =========================
# Train
# =========================
model = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs
)

print("Training completed.")