# infer_utils.py

import torch
from PIL import Image
import torchvision.transforms as transforms
from id_.scripts.u2net.model.u2net import U2NET
import os
import numpy as np

# Load model only once
model_path = r"C:\Users\hp\Desktop\PROJECT\id_card_project\id_\models\u2net.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = U2NET(3, 1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

def predict_mask_and_extract(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)[0]

    # Convert to binary mask
    pred = pred.squeeze().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    pred[pred > 100] = 255
    pred[pred <= 100] = 0
    mask = Image.fromarray(pred)

    # Apply mask
    mask_resized = mask.resize(image.size).convert("L")
    image = image.convert("RGBA")
    result = Image.new("RGBA", image.size)
    result.paste(image, (0, 0), mask_resized)

    return mask, result
