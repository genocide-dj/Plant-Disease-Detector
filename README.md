# 🌿 PlantDoc — AI Plant Disease Detector

> Detect plant diseases instantly from leaf images using deep learning, with treatment advice and Indian government farming policy information.

**Author:** Aman Kumar  
**GitHub:** [genocide-dj/Plant-Disease-Detector](https://github.com/genocide-dj/Plant-Disease-Detector)  
**Model:** [iAmantripathi/plant-disease-detector](https://huggingface.co/iAmantripathi/plant-disease-detector)

---

## 🚀 Live Demo

> Deployment coming soon on Render.com

---

## 📸 Features

- **AI Disease Detection** — Upload a leaf photo and get instant diagnosis across 38 disease classes and 14 plant species
- **99.5% Accuracy** — EfficientNet-B3 model trained on 87,900+ images
- **Grad-CAM Heatmap** — Visualize exactly which part of the leaf the AI focused on
- **Top 3 Predictions** — Confidence scores for the top 3 possible diseases
- **Disease Information** — Symptoms, causes, treatment, and prevention for every disease
- **Plant Encyclopedia** — Dedicated pages for all 14 supported plant species
- **Indian Farming Policies** — Government schemes like PM-KISAN, PMFBY, Kisan Credit Card and more
- **Farmer Helplines** — Official Indian government helpline numbers

---

## 🌱 Supported Plants

| Plant | Conditions Covered |
|-------|--------------------|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust |
| 🍅 Tomato | Early Blight, Late Blight, Leaf Mold, Spider Mites, and more |
| 🥔 Potato | Early Blight, Late Blight |
| 🌽 Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight |
| 🍇 Grape | Black Rot, Esca, Leaf Blight |
| 🍓 Strawberry | Leaf Scorch |
| 🍑 Peach | Bacterial Spot |
| 🫑 Bell Pepper | Bacterial Spot |
| 🍒 Cherry | Powdery Mildew |
| 🍊 Orange | Citrus Greening (HLB) |
| 🫐 Blueberry | Healthy detection |
| 🍓 Raspberry | Healthy detection |
| 🌱 Soybean | Healthy detection |
| 🎃 Squash | Powdery Mildew |

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B3 |
| Training Images | 87,900+ |
| Classes | 38 (37 diseases + healthy) |
| Validation Accuracy | 99.5% |
| Training Platform | Google Colab (T4 GPU) |
| Model Hosting | Hugging Face Hub |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask (Python) |
| Deep Learning | PyTorch + timm |
| Model | EfficientNet-B3 |
| Explainability | Grad-CAM (pytorch-grad-cam) |
| Model Hosting | Hugging Face Hub |
| Frontend | HTML + CSS + Vanilla JS |
| Deployment | Render.com (coming soon) |

---

## 📁 Project Structure

```
Plant-Disease-Detector/
├── app.py                          # Flask backend
├── diseases.py                     # Disease info database
├── class_indices.json              # Class label mappings
├── requirements.txt                # Python dependencies
├── Procfile                        # Render deployment config
├── templates/
│   ├── home.html                   # Home page with upload UI
│   ├── plant.html                  # Individual plant detail page
│   └── policies.html               # Indian government farming policies
└── PlantDiseaseDetector_Training.ipynb   # Model training notebook
```

---

## ⚙️ How It Works

```
User uploads leaf image
        ↓
Flask receives image
        ↓
EfficientNet-B3 classifies disease
        ↓
Grad-CAM generates heatmap
        ↓
Disease info fetched from database
        ↓
Results displayed (disease, confidence, heatmap, treatment)
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11
- Anaconda (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/genocide-dj/Plant-Disease-Detector.git
cd Plant-Disease-Detector

# Create conda environment
conda create -n plantdisease python=3.11 -y
conda activate plantdisease

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open your browser at `http://127.0.0.1:5000`

> The model (~130MB) will be automatically downloaded from Hugging Face on first run.

---

## 📦 Requirements

```
flask
torch
torchvision
timm
opencv-python
pillow
numpy
huggingface-hub
grad-cam
gunicorn
```

---

## 🇮🇳 Indian Farming Policies Covered

- **PM-KISAN** — ₹6,000/year direct income support
- **PMFBY** — Crop insurance at low premiums
- **PM Krishi Sinchai Yojana** — Up to 55% subsidy on irrigation
- **Paramparagat Krishi Vikas Yojana** — ₹50,000/hectare for organic farming
- **Kisan Credit Card** — Credit at 4% interest rate
- **eNAM** — National Agriculture Market for better prices
- **Digital Agriculture Mission** — Drone subsidies and smart farming
- **Soil Health Card Scheme** — Free soil testing every 2 years
- **MIDH** — Up to 100% subsidy for horticulture

---

## 📊 Dataset

**New Plant Diseases Dataset** from Kaggle  
- 87,900 images across 38 classes  
- 14 plant species  
- Train/Validation split included  
- [View Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Aman Kumar**  
- GitHub: [@genocide-dj](https://github.com/genocide-dj)  
- Hugging Face: [@iAmantripathi](https://huggingface.co/iAmantripathi)

---

*Built with ❤️ using PyTorch, Flask, and EfficientNet-B3*
