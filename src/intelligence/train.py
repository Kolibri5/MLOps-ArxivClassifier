import os
import glob
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

def main(n_estimators, max_depth):
    # Setup MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ArXiv_Classification_Experiment")
    
    with mlflow.start_run():
        print(f"\nMemulai Eksperimen | n_estimators: {n_estimators}, max_depth: {max_depth}")
        
        # 1. Load Data (Mengambil data CSV versi terbaru)
        list_of_files = glob.glob('data/processed/*.csv')
        if not list_of_files:
            print("Data tidak ditemukan! Harap jalankan preprocessing dulu.")
            return
        
        latest_file = max(list_of_files, key=os.path.getctime)
        df = pd.read_csv(latest_file)
        
        # Hapus baris yang kosong
        df.dropna(subset=['clean_text', 'primary_category'], inplace=True)
        
        # 2. Split Data
        X = df['clean_text']
        y = df['primary_category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Ekstraksi Fitur (TF-IDF)
        vectorizer = TfidfVectorizer(max_features=1000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # 4. Inisialisasi & Latih Model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train_vec, y_train)
        
        # 5. Evaluasi Metrik
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 6. Logging ke MLflow (Parameter, Metrik, Model)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Selesai! Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

if __name__ == "__main__":
    # Menggunakan argparse agar parameter bisa diubah lewat terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah pohon di Random Forest")
    parser.add_argument("--max_depth", type=int, default=10, help="Kedalaman maksimal pohon")
    args = parser.parse_args()
    
    main(args.n_estimators, args.max_depth)