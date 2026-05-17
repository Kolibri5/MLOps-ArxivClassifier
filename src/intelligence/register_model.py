import mlflow
from mlflow.tracking import MlflowClient

def setup_registry():
    client = MlflowClient()
    model_name = "ArXiv_Classifier_Model"
    
    print("Mencari riwayat eksperimen di MLflow...")
    experiment = client.get_experiment_by_name("ArXiv_Classification_Experiment")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    run_terbaik_id = None
    run_kedua_id = None
    
    # Mencari otomatis run dengan Akurasi 0.4 dan 0.3
    for run in runs:
        acc = run.data.metrics.get("accuracy", 0)
        depth = run.data.params.get("max_depth", "")
        
        if acc == 0.4:
            run_terbaik_id = run.info.run_id
        elif acc == 0.3 and depth == "5":
            run_kedua_id = run.info.run_id
            
    if not run_terbaik_id or not run_kedua_id:
        print("Gagal menemukan ID Run. Pastikan training di LK-06 sudah sukses.")
        return

    print("\nMemulai Proses Model Registry...")
    
    # 1. Registrasi Model Versi 1 (Akurasi 0.4)
    print("Mendaftarkan Versi 1 (Model Terbaik)...")
    mlflow.register_model(f"runs:/{run_terbaik_id}/random_forest_model", model_name)
    
    # 2. Registrasi Model Versi 2 (Akurasi 0.3)
    print("Mendaftarkan Versi 2 (Model Ringan) untuk simulasi versioning...")
    mlflow.register_model(f"runs:/{run_kedua_id}/random_forest_model", model_name)
    
    # 3. Transisi Stage ke Production
    print("Mengubah status Versi 1 menjadi 'Production'...")
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Production"
    )
    
    print("\nPROSES SELESAI! Model siap digunakan untuk Inferensi.")

if __name__ == "__main__":
    setup_registry()