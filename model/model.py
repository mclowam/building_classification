import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output


target_size = (224,224)

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
classes = ("brutalism", "classicism", "modernism")


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3))

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=(3, 3))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(13*13*9, 128)

        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

conv_net = ConvNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conv_net.to(device)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(conv_net.parameters(), lr=learning_rate)


def evaluate(model, data_loader, loss_fn):
    losses =  []

    num_current = 0
    num_elements = 0

    for i, batch in enumerate(data_loader):
        X_batch, y_batch = batch
        with torch.no_grad():
            logits = model(X_batch.to(device))
            loss = loss_fn(logits, y_batch.to(device))
            losses.append(loss.item())

            y_pred = torch.argmax(logits, dim=1).cpu()

            num_current += torch.sum(y_pred == y_batch)
            num_elements += len(X_batch)

        accuracy = num_current / num_elements

        return accuracy, np.mean(losses)

def train(model, loss_fn, optimizer, n_epoch=3):
    num_iter = 0
    log_every = 50

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in tqdm(range(n_epoch)):
        model.train(True)
        print("Epoch: ", epoch)
        epoch_train_losses = []
        epoch_train_acc = []

        for i, batch in enumerate(train_loader):
            X_batch, y_batch = batch
            logits = model(X_batch.to(device))

            loss = loss_fn(logits, y_batch.to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            num_iter += 1

            epoch_train_losses.append(loss.item())

            model_answers = torch.argmax(logits, dim=1).cpu()
            train_accuracy = torch.sum(y_batch == model_answers) / len(y_batch)
            epoch_train_acc.append(train_accuracy)

        model.train(False)

        train_losses.append(np.mean(epoch_train_losses))
        train_acc.append(np.mean(epoch_train_acc))

        val_accuracy, val_loss = evaluate(model=model, data_loader=val_loader, loss_fn=loss_fn)
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)

        clear_output(wait=True)

        _, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].plot(range(len(train_losses)), train_losses, c="b")
        axes[1].plot(range(len(train_acc)), train_acc, c="b")
        axes[0].plot(range(len(val_losses)), val_losses, c="r")
        axes[1].plot(range(len(val_acc)), val_acc, c="r")
        axes[0].set_title("Loss,  epoch done: " + str(epoch))
        axes[1].set_title("Accuracy, epoch done: " + str(epoch))

    return model

conv_net = train(model=conv_net, loss_fn=loss_fn, optimizer=optimizer, n_epoch=10)

X_batch, y_batch = next(iter(test_loader))

with torch.no_grad():
    logits = conv_net(X_batch.to(device))
    preds = torch.argmax(logits, dim=1).cpu().numpy()