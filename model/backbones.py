from torch import nn


def cifar10_backbone(variant):
    layers = [nn.AvgPool2d(2), nn.Conv2d(3, 16, 3, padding=1), nn.ReLU()]
    if variant:
        layers.extend([nn.MaxPool2d(2), nn.Conv2d(16, 24, 3, padding=1), nn.ReLU()])
    layers.extend([nn.AdaptiveAvgPool2d(2), nn.Flatten()])
    return nn.Sequential(*layers), (24 if variant else 16) * 4
