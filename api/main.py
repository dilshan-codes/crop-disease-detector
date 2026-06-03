from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# ── Model Definition ──────────────────────────────────────────
class CropDiseaseCNN(nn.Module):
    def __init__(self, num_classes=15):
        super(CropDiseaseCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


# ── Class Names ───────────────────────────────────────────────
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# ── Load Model ────────────────────────────────────────────────
device = torch.device('cpu')
model = CropDiseaseCNN(num_classes=15)
model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu'))
model.eval()

# ── Transform ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4591, 0.4753, 0.4116],
        std=[0.1812, 0.1573, 0.1956]
    )
])

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="Crop Disease Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "Crop Disease Detector API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top3 = torch.topk(probs, 3)

    return {
        "prediction": CLASS_NAMES[top3.indices[0]],
        "confidence": round(top3.values[0].item() * 100, 2),
        "top3": [
            {
                "disease": CLASS_NAMES[top3.indices[i]],
                "confidence": round(top3.values[i].item() * 100, 2)
            }
            for i in range(3)
        ]
    }