# Image Classification Baseline

This repository trains and evaluates simple image classification models using PyTorch.

The code uses `torchvision.datasets.ImageFolder`, so the dataset must be organised by class folders.

## Repository Structure

```text
classification_repo/
│
├── models.py
├── train.py
├── test.py
├── README.md
│
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
├── test/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
└── checkpoints/