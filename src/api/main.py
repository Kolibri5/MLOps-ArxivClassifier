from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import os
from src.processing.preprocess import clean_text 

# Konfigurasi Koneksi ke MLflow Server
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

app = FastAPI(title="ArXiv Paper Classifier API", version="1.0")

# Skema Input dari User
class PaperInput(BaseModel):
    title: str
    abstract: str

# Variabel global
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"Menghubungkan ke MLflow Tracking Server di: {MLFLOW_URI}")
        
        model_uri = "models:/ArXiv_Classifier_Model@production"
        print(f"Menarik artefak model dari: {model_uri}...")
        
        model = mlflow.pyfunc.load_model(model_uri)
        print("Berhasil! Model @production telah dimuat ke dalam memori API.")
    except Exception as e:
        print(f"CRITICAL ERROR - Gagal memuat model: {e}")
        print("Solusi: Pastikan kontainer mlflow-server menyala, dan model sudah diregister.")

@app.post("/predict")
def predict_category(paper: PaperInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model gagal dimuat saat startup. Cek log kontainer.")
    
    # Feature Engineering & Preprocessing
    full_text = f"{paper.title} {paper.abstract}"
    cleaned_text = clean_text(full_text)
    
    # Prediksi menggunakan Pipeline MLflow
    prediction = model.predict([cleaned_text])
    
    return {
        "title": paper.title,
        "predicted_category": prediction[0],
        "status": "success"
    }

@app.get("/")
def health_check():
    return {"message": "API ArXiv Classifier Aktif dan Siap Menerima Prediksi!"}