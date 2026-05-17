# MLOps-ArxivClassifier

Proyek MLOps untuk klasifikasi paper arXiv dengan pendekatan *Continual Learning*.

---

## Deskripsi Proyek

Proyek ini menggunakan API arXiv untuk mensimulasikan aliran data dinamis sebagai bagian dari implementasi konsep *Continual Learning*. Pipeline mencakup proses pengambilan data terbaru, preprocessing teks, dan persiapan dataset untuk pelatihan model klasifikasi.

---

## Struktur Pipeline ETL

### 1. Data Ingestion

Tahap ini mengambil 50 paper terbaru dari kategori Ilmu Komputer pada arXiv, seperti:

- Artificial Intelligence (AI)
- Computer Vision (CV)
- Machine Learning (ML)

Data mentah akan disimpan pada direktori berikut:

```text
data/raw/
```

Setiap file menggunakan format timestamp untuk menjaga histori data dan mencegah penimpaan file sebelumnya.

#### Menjalankan Proses Ingestion

```bash
python src/ingestion/ingest_data.py
```

---

### 2. Data Preprocessing

Tahap preprocessing bertujuan membersihkan data teks yang berasal dari gabungan judul dan abstrak paper.

Proses yang dilakukan meliputi:

- Menghapus URL
- Menghapus simbol dan karakter khusus
- Menghapus newline
- Normalisasi teks dasar

Hasil preprocessing akan disimpan dalam format CSV pada direktori berikut:

```text
data/processed/
```

Dataset hasil preprocessing siap digunakan untuk proses pelatihan model klasifikasi.

#### Menjalankan Proses Preprocessing

```bash
python src/processing/preprocess.py
```

## Manajemen Versi Data (Data Versioning)

Proyek ini menggunakan **DVC (Data Version Control)** untuk melacak perubahan *dataset* seiring berjalannya waktu (*Continual Learning*).
- File data aktual (`.json`, `.csv`) diabaikan oleh Git (via `.gitignore`) agar repositori tetap ringan.
- DVC menghasilkan file penunjuk (`.dvc`) yang berisi *hash* unik dari *dataset* pada titik waktu tertentu. File `.dvc` inilah yang disimpan di Git.
- Alur pembaruan data: Setiap kali skrip *Ingestion* menarik data baru, jalankan `dvc add data/raw data/processed` diikuti dengan `git commit` pada file `.dvc` yang diperbarui.

## Model Registry & Deployment Readiness
- **Model Aktif:** `ArXiv_Classifier_Model` (Versi 1, Status: Production)
- **Cara Inferensi:** Skrip `src/intelligence/inference.py` memuat model secara dinamis melalui URI `models:/ArXiv_Classifier_Model/Production`.