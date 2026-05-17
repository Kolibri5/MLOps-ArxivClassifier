import mlflow
from mlflow.tracking import MlflowClient
import sys

def evaluate_and_register():
    client = MlflowClient()
    experiment_name = "ArXiv_Classification_Experiment"
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print("Eksperimen tidak ditemukan.")
        sys.exit(1)

    # Ambil run terakhir yang baru saja dieksekusi oleh GitHub Actions
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    latest_run = runs[0]
    accuracy = latest_run.data.metrics.get("accuracy", 0)
    run_id = latest_run.info.run_id
    
    # Menetapkan Ambang Batas (Threshold)
    threshold = 0.45 
    print(f"Mengevaluasi Model... Akurasi Terbaru: {accuracy:.4f} | Threshold: {threshold}")
    
    if accuracy >= threshold:
        print("Evaluasi Sukses! Akurasi di atas threshold. Mendaftarkan ke Registry...")
        model_name = "ArXiv_Classifier_Model"
        
        # Mendaftarkan versi baru
        mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)
        
        # Mencari versi terbaru yang baru saja didaftarkan
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        
        # Transisi ke Staging
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging"
        )
        print(f"Berhasil! Model Versi {latest_version} didaftarkan dengan status 'Staging'.")
    else:
        print("Evaluasi Gagal. Akurasi di bawah threshold. Model dibatalkan.")
        sys.exit(1) # Ini akan menggagalkan pipeline GitHub Actions

if __name__ == "__main__":
    evaluate_and_register()