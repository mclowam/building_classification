import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output


target_size = (224, 224)

transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor()
])
full_data = datasets.ImageFolder(root='./data', transform=transform)

train_size = int(len(full_data) * 0.8)
val_size = len(full_data) - train_size

train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False)
classes = full_data.classes
num_classes = len(classes)


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


def train(model, loss_fn, optimizer, n_epoch=3, save_dir="weights"):
    os.makedirs(save_dir, exist_ok=True)
    best_weights_path = os.path.join(save_dir, "best_model.pth")
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(n_epoch):
        model.train()
        epoch_train_losses = []
        epoch_train_acc = []

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epoch}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

            model_answers = torch.argmax(logits, dim=1)
            train_accuracy = torch.sum(y_batch == model_answers).item() / len(y_batch)
            epoch_train_acc.append(train_accuracy)

        train_losses.append(np.mean(epoch_train_losses))
        train_acc.append(np.mean(epoch_train_acc))

        val_accuracy, val_loss = evaluate(model=model, data_loader=val_loader, loss_fn=loss_fn)
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)

        log_msg = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": full_data.class_to_idx,
                "target_size": target_size,
            }, best_weights_path)
            log_msg = f"loss уменьшилось: {val_loss}, веса обновлены"

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            os.path.join(save_dir, "last_checkpoint.pth"),
        )

        clear_output(wait=True)

        if log_msg:
            print(log_msg)

        # fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        # axes[0].plot(range(len(train_losses)), train_losses, c="b", label="Train")
        # axes[0].plot(range(len(val_losses)), val_losses, c="r", label="Val")
        # axes[0].set_title(f"Loss, epoch done: {epoch + 1}")
        # axes[0].legend()
        #
        # axes[1].plot(range(len(train_acc)), train_acc, c="b", label="Train")
        # axes[1].plot(range(len(val_acc)), val_acc, c="r", label="Val")
        # axes[1].set_title(f"Accuracy, epoch done: {epoch + 1}")
        # axes[1].legend()

    return model


conv_net = ConvNet(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv_net.to(device)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(conv_net.parameters(), lr=learning_rate)

best_weights_path = os.path.join("weights", "best_model.pth")
last_checkpoint_path = os.path.join("weights", "last_checkpoint.pth")

if os.path.exists(last_checkpoint_path):
    try:
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        conv_net.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Возобновление с эпохи {checkpoint['epoch']}, оптимизатор восстановлен")
    except RuntimeError:
        print("обучение с нуля")

conv_net = train(model=conv_net, loss_fn=loss_fn, optimizer=optimizer, n_epoch=10)

if os.path.exists(best_weights_path):
    try:
        best_checkpoint = torch.load(best_weights_path, map_location=device)
        conv_net.load_state_dict(best_checkpoint["model_state"])
    except RuntimeError:
        pass

conv_net.eval()

X_batch, y_batch = next(iter(val_loader))

with torch.no_grad():
    logits = conv_net(X_batch.to(device))
    preds = torch.argmax(logits, dim=1).cpu().numpy()

fig, axes = plt.subplots(2, 4, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    img = X_batch[i].numpy()
    img = np.transpose(img, (1, 2, 0))

    ax.imshow(img)

    pred_class = classes[preds[i]]
    true_class = classes[y_batch[i].item()]

    color = "green" if preds[i] == y_batch[i].item() else "red"
    ax.set_title(f"Pred: {pred_class}\n(True: {true_class})", color=color)
    ax.axis("off")
plt.tight_layout()
plt.show()