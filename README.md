# 🌿 Crop Disease Detector

A deep learning web application that classifies plant leaf diseases from images.
Built with a CNN trained **completely from scratch** — no pretrained weights used.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)

---

## Results

| Metric | Score |
|---|---|
| Validation Accuracy | 98.01% |
| Test Accuracy | 96.90% |
| Classes | 15 |
| Training Images | 16,510 |
| Parameters | 26M |

---

## Disease Classes

The model detects 15 plant diseases across 3 crops:

**Tomato** — Bacterial Spot, Early Blight, Late Blight, Leaf Mold,
Septoria Leaf Spot, Spider Mites, Target Spot, YellowLeaf Curl Virus,
Mosaic Virus, Healthy

**Potato** — Early Blight, Late Blight, Healthy

**Pepper** — Bacterial Spot, Healthy

---

## Project Structure

```
crop-disease-detector/
├── api/
│   └── main.py          → FastAPI backend (prediction endpoint)
├── notebooks/
│   └── 01_eda.ipynb     → Full training notebook (Colab)
├── src/
│   ├── model.py         → CNN architecture
│   ├── dataset.py       → Data loading, transforms, splits
│   ├── train.py         → Training loop
│   └── evaluate.py      → Evaluation, confusion matrix
├── ui/
│   └── index.html       → Frontend web interface
├── models/              → Model checkpoint (not in repo — see below)
└── requirements.txt
```

---

## Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/dilshan-codes/crop-disease-detector.git
cd crop-disease-detector
```

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the model

Download `best_model.pth` from Google Drive and place it in the `models/` folder:

```
models/
└── best_model.pth
```

> 📎 [Download best_model.pth](YOUR_GOOGLE_DRIVE_LINK_HERE)

### 5. Start the API

```bash
uvicorn api.main:app --reload
```

API runs at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### 6. Open the UI

Open `ui/index.html` directly in your browser.
Upload a leaf image and click **Analyze Leaf**.

---

## Model Architecture

4-block CNN built from scratch in PyTorch:

```
Input (224x224x3)
    ↓
Block 1: Conv(3→32)   → BatchNorm → ReLU → MaxPool  → 112x112x32
Block 2: Conv(32→64)  → BatchNorm → ReLU → MaxPool  → 56x56x64
Block 3: Conv(64→128) → BatchNorm → ReLU → MaxPool  → 28x28x128
Block 4: Conv(128→256)→ BatchNorm → ReLU → MaxPool  → 14x14x256
    ↓
Flatten → Dropout(0.5) → Linear(512) → ReLU → Dropout(0.3) → Linear(15)
    ↓
Softmax → 15 class probabilities
```

---

## Training Details

| Setting | Value |
|---|---|
| Dataset | PlantVillage (20,639 images) |
| Split | 80% train / 10% val / 10% test |
| Optimizer | Adam (lr=0.001) |
| Loss | CrossEntropyLoss with class weights |
| Epochs | 50 |
| Batch size | 32 |
| GPU | Google Colab T4 |
| Training time | ~2 hours |

**Class imbalance** handled with weighted loss function.
Weights computed as `total_samples / (num_classes × class_count)`.

---

## Tech Stack

- **PyTorch** — model definition and training
- **torchvision** — image transforms and data loading
- **FastAPI** — REST API backend
- **Uvicorn** — ASGI server
- **scikit-learn** — evaluation metrics
- **Pillow** — image processing
- **HTML/CSS/JS** — frontend (no frameworks)

---

## API Endpoints

### `POST /predict`
Upload a leaf image and get disease prediction.

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "prediction": "Tomato_Late_blight",
  "confidence": 94.32,
  "top3": [
    { "disease": "Tomato_Late_blight", "confidence": 94.32 },
    { "disease": "Tomato_Early_blight", "confidence": 3.21 },
    { "disease": "Tomato_Bacterial_spot", "confidence": 1.15 }
  ]
}
```

### `GET /`
Health check — returns API status.

---

## Retrain the Model

Open `notebooks/01_eda.ipynb` in Google Colab:
- Runtime → Change runtime type → T4 GPU
- Add your Kaggle API token
- Run all cells
- Model saves to Google Drive automatically