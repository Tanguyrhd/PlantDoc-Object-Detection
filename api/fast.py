from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
from pathlib import Path
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import cv2
import numpy as np
import base64

app = FastAPI()

# Charger les trois modèles au démarrage de l'application
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "results"

def load_models():
    """Charge les trois modèles YOLO au démarrage"""
    models = {
        'binary': YOLO(str(MODELS_DIR / "binary" / "models" / "best.pt")),
        'species': YOLO(str(MODELS_DIR / "species" / "models" / "best.pt")),
        'diseases': YOLO(str(MODELS_DIR / "diseases" / "models" / "best.pt"))
    }
    return models

# Stocker les modèles dans l'état de l'application
app.state.models = load_models()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get('/')
def root():
    return {
        'message': 'PlantDoc API',
        'models_loaded': list(app.state.models.keys())
    }

async def predict_with_model(file: UploadFile, model: YOLO):
    """Fonction générique pour faire des prédictions avec n'importe quel modèle"""
    # Temp image save
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    # Predict
    results = model.predict(temp_path)
    result = results[0]

    # Get annotated image as base64
    annotated_img = result.plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_img)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()

    # Get predictions details
    predictions = []
    if result.boxes:
        for box in result.boxes:
            predictions.append({
                'class_id': int(box.cls[0]),
                'class_name': result.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            })

    # Nettoyer le fichier temporaire
    os.remove(temp_path)

    return {
        'predictions': predictions,
        'annotated_image': f"data:image/png;base64,{img_base64}"
    }

@app.post('/predict/binary')
async def predict_binary(file: UploadFile = File(...)):
    """Prédiction binaire : plante saine ou malade"""
    return await predict_with_model(file, app.state.models['binary'])

@app.post('/predict/species')
async def predict_species(file: UploadFile = File(...)):
    """Prédiction de l'espèce de plante"""
    return await predict_with_model(file, app.state.models['species'])

@app.post('/predict/diseases')
async def predict_diseases(file: UploadFile = File(...)):
    """Prédiction des maladies de plante"""
    return await predict_with_model(file, app.state.models['diseases'])
