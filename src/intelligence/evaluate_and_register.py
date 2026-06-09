import mlflow
from mlflow.tracking import MlflowClient
import sys

def evaluate_and_register():
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    experiment_name = "ArXiv_Classification_Experiment"
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print("Eksperimen tidak ditemukan.")
        sys.exit(1)

    # Ambil run terbaru (Challenger)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    latest_run = runs[0]
    new_accuracy = latest_run.data.metrics.get("accuracy", 0)
    run_id = latest_run.info.run_id
    
    # Ambil akurasi model saat ini (Champion)
    model_name = "ArXiv_Classifier_Model"
    champion_accuracy = 0.0
    try:
        prod_model = client.get_model_version_by_alias(model_name, "production")
        prod_run = client.get_run(prod_model.run_id)
        champion_accuracy = prod_run.data.metrics.get("accuracy", 0.0)
        print(f"Model Champion (@production) ditemukan! Akurasi Saat Ini: {champion_accuracy:.4f}")
    except Exception as e:
        print("Model @production belum ada atau gagal diambil. Menggunakan threshold dasar 0.75")
        champion_accuracy = 0.75

    print(f"Mengevaluasi Challenger (Baru: {new_accuracy:.4f}) vs Champion (Lama: {champion_accuracy:.4f})")
    
    # Logika Promosi Bersyarat
    if new_accuracy > champion_accuracy:
        print("Evaluasi Sukses! Model baru LEBIH BAIK. Mendaftarkan ke Registry...")
        
        logged_models = client.search_logged_models(experiment_ids=[experiment.experiment_id])
        run_model = next((lm for lm in logged_models if lm.source_run_id == run_id), None)
        
        if not run_model:
            print(f"Error: Logged Model tidak ditemukan untuk Run ID: {run_id}")
            sys.exit(1)
            
        model_version_info = mlflow.register_model(run_model.model_uri, model_name)
        client.set_registered_model_alias(name=model_name, alias="production", version=model_version_info.version)
        print(f"Berhasil! Model Versi {model_version_info.version} resmi menjadi @production yang baru.")
    else:
        print("Evaluasi Selesai. Model baru TIDAK MENGALAHKAN model saat ini. Pipeline dihentikan tanpa promosi.")
        sys.exit(0) 

if __name__ == "__main__":
    evaluate_and_register()