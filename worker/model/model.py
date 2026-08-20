import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from worker.model.settings import device


class ConvNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluate(model, data_loader, loss_fn):
    losses = []
    num_current = 0
    num_elements = 0

    model.eval()
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        with torch.no_grad():
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            losses.append(loss.item())

            y_pred = torch.argmax(logits, dim=1)
            num_current += torch.sum(y_pred == y_batch).item()
            num_elements += len(X_batch)

    accuracy = num_current / num_elements
    return accuracy, np.mean(losses)
