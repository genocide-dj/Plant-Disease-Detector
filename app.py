from flask import Flask, render_template, request, jsonify, redirect, url_for
import torch
import torchvision.transforms as transforms
import timm
from PIL import Image
import numpy as np
import cv2
import base64
import json
import io
import os
from huggingface_hub import hf_hub_download
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from diseases import PLANTS, get_disease_info

app = Flask(__name__)

# ── Load class names ──────────────────────────────────────────
with open("class_indices.json") as f:
    class_indices = json.load(f)
idx_to_class = {int(k): v for k, v in class_indices.items()}
NUM_CLASSES = len(idx_to_class)

# ── Load model ────────────────────────────────────────────────
print("Loading model from Hugging Face...")
model_path = hf_hub_download(
    repo_id="iAmantripathi/plant-disease-detector",
    filename="best_model.pth"
)
model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=NUM_CLASSES)
checkpoint = torch.load(model_path, map_location="cpu")
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()
print("Model loaded!")

# ── Transform ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Grad-CAM ──────────────────────────────────────────────────
def generate_gradcam(img_pil, input_tensor):
    target_layer = model.conv_head
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0))[0]
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html", plants=PLANTS)

@app.route("/policies")
def policies():
    return render_template("policies.html")

@app.route("/plant/<plant_key>")
def plant_detail(plant_key):
    plant = PLANTS.get(plant_key)
    if not plant:
        return redirect(url_for("home"))
    return render_template("plant.html", plant=plant, plant_key=plant_key)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    input_tensor = transform(img)

    with torch.no_grad():
        outputs = model(input_tensor.unsqueeze(0))
        probs = torch.softmax(outputs, dim=1)[0]
        top3 = torch.topk(probs, 3)

    predictions = []
    for i in range(3):
        idx = top3.indices[i].item()
        conf = top3.values[i].item() * 100
        class_name = idx_to_class[idx]
        info = get_disease_info(class_name)
        predictions.append({
            "class": class_name,
            "display_name": f"{info['plant']} - {info['disease_name']}",
            "confidence": round(conf, 2)
        })

    top_class = idx_to_class[top3.indices[0].item()]
    disease_info = get_disease_info(top_class)
    gradcam_img = generate_gradcam(img, input_tensor)

    # Convert original image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    original_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        "predictions": predictions,
        "gradcam": gradcam_img,
        "original": original_b64,
        "disease_info": disease_info
    })

if __name__ == "__main__":
    app.run(debug=True)