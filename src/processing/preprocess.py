import pandas as pd
import glob
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Inisialisasi alat NLP
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Basic Cleaning
    text = text.lower() # Case folding
    text = re.sub(r'http\S+|www\.\S+', '', text) # Hapus URL jika ada
    text = re.sub(r'\n', ' ', text) # Ganti enter/newline dengan spasi
    text = re.sub(r'[^a-z0-9\s]', ' ', text) # Hapus simbol aneh, sisakan huruf dan angka
    
    # 2. Stopwords Removal & Lemmatization
    words = text.split()
    # Simpan kata JIKA bukan stopword, lalu ubah ke kata dasar (lemmatize)
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Gabungkan kembali menjadi satu kalimat
    text = ' '.join(cleaned_words)
    
    # 3. Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_preprocessing():
    print("Memulai proses Preprocessing tingkat lanjut (NLP)...")
    
    # Cari file JSON terbaru di data/raw/
    list_of_files = glob.glob('data/raw/*.json')
    if not list_of_files:
        print("Tidak ada data mentah ditemukan!")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Memproses file: {latest_file}")
    
    df = pd.read_json(latest_file)
    
    # Cek missing values
    df.fillna({'title': '', 'abstract': '', 'primary_category': 'unknown'}, inplace=True)
    
    # FEATURE ENGINEERING: Menggabungkan Judul dan Abstrak
    df['full_text'] = df['title'] + " " + df['abstract']
    
    # Membersihkan teks gabungan tersebut (Ini akan memakan waktu sedikit lebih lama karena Lemmatization)
    print("Sedang membersihkan teks dan melakukan Lemmatization (ini mungkin butuh beberapa saat)...")
    df['clean_text'] = df['full_text'].apply(clean_text)
    
    # Memilih kolom yang relevan
    df_final = df[['paper_id', 'published_date', 'primary_category', 'clean_text']]
    
    # Menyimpan hasil ke data/processed/
    base_name = os.path.basename(latest_file).replace('.json', '')
    os.makedirs("data/processed", exist_ok=True)
    output_path = f"data/processed/clean_{base_name}.csv"
    
    df_final.to_csv(output_path, index=False)
    print(f"Preprocessing Selesai! Data bersih dengan {len(df_final)} baris disimpan di {output_path}")

if __name__ == "__main__":
    run_preprocessing()