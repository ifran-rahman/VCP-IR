import torch.nn as nn
from torchvision import models


def get_model(model_name, num_classes):
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            "Choose from: resnet18, resnet50, efficientnet_b0"
        )

    return model