import mlflow
from mlflow.tracking import MlflowClient
import sys

def evaluate_and_register():
    # Koneksi ke Database SQLite
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    experiment_name = "ArXiv_Classification_Experiment"
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print("Eksperimen tidak ditemukan.")
        sys.exit(1)

    # Ambil run terakhir
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    latest_run = runs[0]
    accuracy = latest_run.data.metrics.get("accuracy", 0)
    run_id = latest_run.info.run_id
    
    threshold = 0.75 
    print(f"Mengevaluasi Model... Akurasi Terbaru: {accuracy:.4f} | Threshold: {threshold}")
    
    if accuracy >= threshold:
        print("Evaluasi Sukses! Mendaftarkan ke Registry...")
        model_name = "ArXiv_Classifier_Model"
        
        # MLflow 3.x: Model disimpan sebagai Logged Model, bukan di run artifact path.
        # Cari Logged Model yang terkait dengan run ini.
        logged_models = client.search_logged_models(experiment_ids=[experiment.experiment_id])
        run_model = None
        for lm in logged_models:
            if lm.source_run_id == run_id:
                run_model = lm
                break
        
        if not run_model:
            print(f"Error: Logged Model tidak ditemukan untuk Run ID: {run_id}")
            sys.exit(1)
        
        print(f"Ditemukan Logged Model: {run_model.model_uri}")
        
        # Mendaftarkan model menggunakan URI Logged Model
        model_version_info = mlflow.register_model(run_model.model_uri, model_name)
        
        client.set_registered_model_alias(
            name=model_name, 
            alias="production", 
            version=model_version_info.version
        )
        
        print(f"Berhasil! Pipeline Model Versi {model_version_info.version} diberi alias '@production'.")
    else:
        print("Evaluasi Gagal. Akurasi di bawah threshold.")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_and_register()