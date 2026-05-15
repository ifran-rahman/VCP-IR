import os
import random
import numpy as np

import torch
import torch.nn as nn

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
batch_size = 1
image_size = 128

ckpt_path = os.path.join(checkpoint_dir, f"{model_name}.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# Transforms
# =========================
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])


# =========================
# Dataset and DataLoader
# =========================
train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform
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
print("Testing images:", len(test_dataset))


# =========================
# Model
# =========================
model = get_model(model_name, num_classes)
model = model.to(device)

print(f"Loaded model architecture: {model_name}")
print(f"Checkpoint path: {ckpt_path}")


# =========================
# Loss
# =========================
criterion = nn.CrossEntropyLoss()


# =========================
# Test Function
# =========================
def test_model(model, test_loader, criterion, device, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the model first using model_name='{model_name}'."
        )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint from: {ckpt_path}")

    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_preds = []
    all_labels = []

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

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= test_total
    test_acc = test_correct / test_total

    print("\n===== Test Results =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.4f}")

    return test_loss, test_acc, all_preds, all_labels


# =========================
# Run Test
# =========================
test_loss, test_acc, preds, labels = test_model(
    model=model,
    test_loader=test_loader,
    criterion=criterion,
    device=device,
    ckpt_path=ckpt_path
)