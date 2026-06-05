---
title: Crop Disease Detector
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
pinned: false
---

# 🌿 Crop Disease Detector

A deep learning web application that classifies plant leaf diseases from images.
Built with a CNN trained **completely from scratch** — no pretrained weights used.

<table>
  <tr>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/8434330b-d975-4d1b-adad-b511112d5e98" width="100%">
    </td>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/4bcc39e9-9680-458f-b460-445821796685" width="100%">
    </td>
  </tr>
</table>

<table>
  <tr>
    <td width="25%">
      <img src="https://github.com/user-attachments/assets/cbb3984d-d148-4d43-8d79-79b7b10ccf16" width="100%">
    </td>
    <td width="25%">
      <img src="https://github.com/user-attachments/assets/4465eae7-2c12-4e86-80a9-94a1f8c3d0bd" width="100%">
    </td>
    <td width="25%">
      <img src="https://github.com/user-attachments/assets/097f19fe-ba74-4f84-aadb-7854907a44c1" width="100%">
    </td>
    <td width="25%">
      <img src="https://github.com/user-attachments/assets/2759252c-1855-439b-8297-b00edac71667" width="100%">
    </td>
  </tr>
</table>

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-yellow)

🚀 **[Live Demo](https://huggingface.co/spaces/Dilshna/crop-disease-detector)**

---

## Results

| Metric | Score |
|---|---|
| Validation Accuracy | 98.01% |
| Test Accuracy | 96.90% |
| Classes | 15 |
| Training Images | 16,510 |
| Model Parameters | 26M |
| Training Time | ~2 hours (Colab T4) |

---

## Disease Classes

The model detects 15 plant diseases across 3 crops:

**Tomato (10 classes)** — Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, YellowLeaf Curl Virus, Mosaic Virus, Healthy

**Potato (3 classes)** — Early Blight, Late Blight, Healthy

**Pepper (2 classes)** — Bacterial Spot, Healthy

---

## Project Structure

```
crop-disease-detector/
├── api/
│   └── main.py          → FastAPI backend + serves frontend
├── notebooks/
│   └── 01_eda.ipynb     → EDA + training notebook (Google Colab)
├── src/
│   ├── model.py         → CNN architecture definition
│   ├── dataset.py       → Data loading, transforms, train/val/test splits
│   ├── train.py         → Training loop with class weights
│   └── evaluate.py      → Confusion matrix and classification report
├── ui/
│   └── index.html       → Frontend web interface
├── models/              → Model checkpoint (not in repo — download below)
├── Dockerfile           → Container config for HuggingFace deployment
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

# Windows
.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the model

Download `best_model.pth` and place it in the `models/` folder:

> 📎 [Download best_model.pth from Google Drive](https://drive.google.com/file/d/1WD09SmKPGcx0rW3lxtoQNu3sAk7HSjrn/view?usp=sharing)

```
models/
└── best_model.pth
```

### 5. Start the API

```bash
uvicorn api.main:app --reload
```

- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

### 6. Open the UI

Open `ui/index.html` in your browser, upload a leaf image and click **Analyze Leaf**.

---

## Model Architecture

4-block CNN built entirely from scratch in PyTorch:

```
Input Image (224 × 224 × 3)
        ↓
Block 1: Conv(3→32)    → BatchNorm → ReLU → MaxPool  →  112×112×32
Block 2: Conv(32→64)   → BatchNorm → ReLU → MaxPool  →  56×56×64
Block 3: Conv(64→128)  → BatchNorm → ReLU → MaxPool  →  28×28×128
Block 4: Conv(128→256) → BatchNorm → ReLU → MaxPool  →  14×14×256
        ↓
Flatten → Dropout(0.5) → Linear(512) → ReLU → Dropout(0.3) → Linear(15)
        ↓
15 class probabilities
```

**Key design decisions:**
- BatchNorm after every Conv layer for training stability
- Dropout(0.5) and Dropout(0.3) in classifier to prevent overfitting
- CrossEntropyLoss with class weights to handle imbalanced dataset
- Adam optimizer with lr=0.001

---

## Training Details

| Setting | Value |
|---|---|
| Dataset | PlantVillage (20,639 images) |
| Split | 80% train / 10% val / 10% test |
| Optimizer | Adam (lr=0.001) |
| Loss | CrossEntropyLoss + class weights |
| Epochs | 50 |
| Batch size | 32 |
| GPU | Google Colab T4 |

**Class imbalance** — dataset ranged from 152 images (Potato Healthy) to 3,209 images (Tomato YellowLeaf Curl). Handled using weighted loss:

```
weight = total_samples / (num_classes × class_count)
```

Minority classes received up to 9× higher penalty for mistakes.

---

## API Reference

### `POST /predict`

Upload a leaf image, get disease prediction back.

**Request:** `multipart/form-data` with `file` field (JPG or PNG)

**Response:**
```json
{
  "prediction": "Tomato_Late_blight",
  "confidence": 94.32,
  "top3": [
    { "disease": "Tomato_Late_blight",    "confidence": 94.32 },
    { "disease": "Tomato_Early_blight",   "confidence": 3.21  },
    { "disease": "Tomato_Bacterial_spot", "confidence": 1.15  }
  ]
}
```

### `GET /`

Returns the frontend HTML UI.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch (CNN from scratch) |
| Data | torchvision, Pillow |
| API | FastAPI + Uvicorn |
| Frontend | HTML / CSS / Vanilla JS |
| Evaluation | scikit-learn |
| Training | Google Colab T4 GPU |
| Deployment | Hugging Face Spaces (Docker) |

---

## Retrain from Scratch

1. Open `notebooks/01_eda.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Add your Kaggle API token in the dataset cell
4. Run all cells — training takes ~2 hours
5. Model checkpoint saves automatically to Google Drive
