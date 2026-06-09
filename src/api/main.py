from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import mlflow
import os
import urllib.request
import json
from src.processing.preprocess import clean_text 
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

# Konfigurasi Koneksi ke MLflow Server
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

app = FastAPI(title="ArXiv Paper Classifier API", version="1.0")

PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total prediksi model berdasarkan kategori",
    ["predicted_category"]
)

class PaperInput(BaseModel):
    title: str
    abstract: str

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"Menghubungkan ke MLflow Tracking Server di: {MLFLOW_URI}")
        model_uri = "models:/ArXiv_Classifier_Model@production"
        model = mlflow.pyfunc.load_model(model_uri)
        print("Berhasil! Model @production telah dimuat.")
    except Exception as e:
        print(f"ERROR: {e}")

@app.post("/predict")
def predict_category(paper: PaperInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model gagal dimuat.")
    
    full_text = f"{paper.title} {paper.abstract}"
    cleaned_text = clean_text(full_text)
    prediction = model.predict([cleaned_text])
    predicted_cat = prediction[0]
    
    PREDICTION_COUNTER.labels(predicted_category=predicted_cat).inc()
    return {"title": paper.title, "predicted_category": predicted_cat, "status": "success"}

@app.get("/")
def health_check():
    return {"message": "API ArXiv Classifier Aktif!"}

@app.post("/webhook/grafana-to-github")
async def grafana_webhook_forwarder(request: Request, background_tasks: BackgroundTasks):
    def send_dispatch():
        github_url = os.getenv("GITHUB_DISPATCH_URL")
        github_pat = os.getenv("GITHUB_PAT")
        
        if not github_pat or not github_url:
            print("ERROR: GITHUB_PAT atau GITHUB_DISPATCH_URL tidak ditemukan di Environment!")
            return

        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {github_pat}", 
            "Content-Type": "application/json"
        }
        
        data = json.dumps({"event_type": "grafana-alert-drift"}).encode("utf-8")
        req = urllib.request.Request(github_url, data=data, headers=headers, method="POST")
        try:
            urllib.request.urlopen(req)
            print("SUKSES: Alarm Grafana diteruskan ke GitHub Actions!")
        except Exception as e:
            print(f"GAGAL: Terjadi kesalahan saat menghubungi GitHub: {e}")

    background_tasks.add_task(send_dispatch)
    return {"status": "Alarm diterima oleh FastAPI, sedang diproses ke GitHub..."}

Instrumentator().instrument(app).expose(app)