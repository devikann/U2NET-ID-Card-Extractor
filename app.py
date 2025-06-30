import os
import sys
sys.path.append('./scripts')

from flask import Flask, render_template, request, send_file
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from u2net.model.u2net import U2NET  # Adjust if needed

app = Flask(__name__)

# Make sure folders exist inside static
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = U2NET(3, 1)
model.load_state_dict(torch.load('models/u2net.pth', map_location=device))
model.to(device)
model.eval()

# Preprocess and postprocess
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def postprocess(mask, original_size):
    mask = mask.squeeze().cpu().detach().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, original_size)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/second')
def second():
    return render_template('second.html')
@app.route('/upload', methods=['POST'])
def upload():
    filename = ""

    # Handle webcam capture
    if 'captured_image' in request.form:
        import base64
        img_data = request.form['captured_image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        filename = 'captured.jpg'
        path = os.path.join(UPLOAD_FOLDER, filename)
        with open(path, 'wb') as f:
            f.write(img_bytes)

    # Handle file upload
    elif 'image' in request.files:
        file = request.files['image']
        if not file or file.filename == '':
            return 'No file uploaded', 400
        filename = file.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
    else:
        return 'No valid image input found.', 400

    # Process the image
    orig_image = Image.open(path).convert('RGB')
    input_tensor = preprocess(orig_image).to(device)

    with torch.no_grad():
        d1 = model(input_tensor)[0]
        mask = torch.sigmoid(d1)

    binary_mask = postprocess(mask, orig_image.size)
    orig_cv = np.array(orig_image)
    masked = cv2.bitwise_and(orig_cv, orig_cv, mask=binary_mask)

    rgba = cv2.cvtColor(masked, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = binary_mask
    pil_rgba = Image.fromarray(rgba)
    rgb_image = pil_rgba.convert('RGB')

    extracted_filename = 'output.jpg'
    out_path = os.path.join(OUTPUT_FOLDER, extracted_filename)
    rgb_image.save(out_path)

    return render_template('second.html', uploaded_img=filename, extracted_img=extracted_filename)

if __name__ == '__main__':
    app.run(debug=True)
