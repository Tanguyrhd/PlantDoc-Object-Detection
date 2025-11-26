from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
from pathlib import Path

app = FastAPI()

# Charger les trois modèles au démarrage de l'application
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "results"

def load_models():
    """Charge les trois modèles YOLO au démarrage"""
    models = {
        'binary': YOLO(str(MODELS_DIR / "binary" / "models" / "best_v0.pt")),
        'species': YOLO(str(MODELS_DIR / "species" / "models" / "best_v0.pt")),
        'diseases': YOLO(str(MODELS_DIR / "diseases" / "models" / "best_v0.pt"))
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

@app.post('/predict/binary')
async def predict_binary(file: UploadFile = File(...)):
    """Prédiction binaire : plante saine ou malade"""
    model = app.state.models['binary']

    # Sauvegarder temporairement l'image
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    # Faire la prédiction
    results = model.predict(temp_path)

    # Nettoyer le fichier temporaire
    os.remove(temp_path)

    return {'predictions': results[0].tojson()}

@app.post('/predict/species')
async def predict_species(file: UploadFile = File(...)):
    """Prédiction de l'espèce de plante"""
    model = app.state.models['species']

    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    results = model.predict(temp_path)
    os.remove(temp_path)

    return {'predictions': results[0].tojson()}

@app.post('/predict/diseases')
async def predict_diseases(file: UploadFile = File(...)):
    """Prédiction des maladies de plante"""
    model = app.state.models['diseases']

    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    results = model.predict(temp_path)
    os.remove(temp_path)

    return {'predictions': results[0].tojson()}
