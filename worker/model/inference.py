import os
import torch
from PIL import Image
from worker.model.model import ConvNet
from torchvision import transforms
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
best_weights_path = os.path.join(BASE_DIR, "weights", "best_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(best_weights_path):
    raise FileNotFoundError(f"Файла весов нет: {best_weights_path}")

try:
    best_weights = torch.load(best_weights_path, map_location=device)
except RuntimeError as e:
    raise RuntimeError(f"Не удалось загрузить веса: {e}")

class_to_idx = best_weights["class_to_idx"]
idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
target_size = best_weights["target_size"]

conv_net = ConvNet(num_classes=len(class_to_idx))
conv_net.load_state_dict(best_weights["model_state"])
conv_net.to(device)
conv_net.eval()

transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor()
])


def inference_predict_style(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = conv_net(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return {
        "style": idx_to_class[pred_idx],
        "confidence": confidence
    }