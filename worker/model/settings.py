from torchvision import transforms, datasets
import torch


target_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
