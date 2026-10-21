import os

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from worker.model.model import ConvNet, evaluate
from worker.model.settings import train_loader, val_loader, full_data, target_size, num_classes, classes, device
import matplotlib.pyplot as plt


weights_dir = os.path.join(os.path.dirname(__file__), "weights")


def train(model, loss_fn, optimizer, n_epoch=3, save_dir=weights_dir):
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

        if log_msg:
            print(log_msg)


    return model


def main():
    conv_net = ConvNet(num_classes=num_classes)
    conv_net.to(device)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(conv_net.parameters(), lr=learning_rate)

    best_weights_path = os.path.join(weights_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(weights_dir, "last_checkpoint.pth")

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


if __name__ == "__main__":
    main()
