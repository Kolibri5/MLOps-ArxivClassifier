import mlflow.pyfunc

def test_inference():
    print("Memulai simulasi inferensi...")
    model_uri = "models:/ArXiv_Classifier_Model/Production"
    print(f"Mencoba memuat model dari URI: {model_uri}")
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("SUKSES! Model Production berhasil dimuat dan siap melayani prediksi.")
    except Exception as e:
        print(f"Gagal memuat model: {e}")

if __name__ == "__main__":
    test_inference()