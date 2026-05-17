import pandas as pd
import glob
import os
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() # Case folding
    text = re.sub(r'http\S+|www\.\S+', '', text) # Hapus URL jika ada
    text = re.sub(r'\n', ' ', text) # Ganti enter/newline dengan spasi
    text = re.sub(r'[^a-z0-9\s-]', '', text) # Hapus simbol aneh, sisakan huruf, angka, spasi, dan strip
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
    return text

def run_preprocessing():
    print("Memulai proses Preprocessing...")
    
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
    
    # Membersihkan teks gabungan tersebut
    df['clean_text'] = df['full_text'].apply(clean_text)
    
    # Memilih kolom yang relevan untuk proses Machine Learning (Text dan Label)
    df_final = df[['paper_id', 'published_date', 'primary_category', 'clean_text']]
    
    # Menyimpan hasil ke data/processed/
    base_name = os.path.basename(latest_file).replace('.json', '')
    os.makedirs("data/processed", exist_ok=True)
    output_path = f"data/processed/clean_{base_name}.csv"
    
    df_final.to_csv(output_path, index=False)
    print(f"Preprocessing Selesai! Data siap klasifikasi disimpan di {output_path}")

if __name__ == "__main__":
    run_preprocessing()