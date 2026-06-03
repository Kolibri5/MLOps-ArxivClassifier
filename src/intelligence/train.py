import os
import glob
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

def main(n_estimators, max_depth):
    # Setup MLflow tracking (Menggunakan SQLite)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("ArXiv_Classification_Experiment")
    
    actual_max_depth = None if max_depth == 0 else max_depth
    
    with mlflow.start_run():
        print(f"\nMemulai Eksperimen | n_estimators: {n_estimators}, max_depth: {actual_max_depth}")
        
        # 1. Load Data (Mengambil data CSV versi terbaru)
        list_of_files = glob.glob('data/processed/*.csv')
        if not list_of_files:
            print("Data tidak ditemukan! Harap jalankan preprocessing dulu.")
            return
        
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Menggunakan data: {latest_file}")
        df = pd.read_csv(latest_file)
        
        df.dropna(subset=['clean_text', 'primary_category'], inplace=True)
        
        # 2. Split Data
        X = df['clean_text']
        y = df['primary_category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3 & 4. Ekstraksi Fitur (TF-IDF) + Model menjadi satu kesatuan
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=actual_max_depth, 
                random_state=42, 
                n_jobs=-1
            ))
        ])
        
        print("Sedang melatih model (menggunakan N-Grams dan Multi-core CPU)...")
        pipeline.fit(X_train, y_train)
        
        # 5. Evaluasi Metrik
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 6. Logging ke MLflow (Simpan Parameter, Metrik, dan Pipeline Model utuh)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", actual_max_depth)
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "(1, 2)")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Simpan Pipeline Utuh (TF-IDF + Random Forest)
        mlflow.sklearn.log_model(pipeline, name="arxiv_classification_pipeline")
        
        print(f"\nSelesai! Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        print("Model Pipeline dan TF-IDF Vectorizer berhasil disimpan ke MLflow!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah pohon di Random Forest")
    parser.add_argument("--max_depth", type=int, default=0, help="Kedalaman maksimal pohon (Ketik 0 untuk kedalaman tak terbatas)")
    args = parser.parse_args()
    
    main(args.n_estimators, args.max_depth)