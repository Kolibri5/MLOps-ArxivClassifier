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

## Cara Menjalankan Sistem (Deployment)

Sistem ini telah dikontainerisasi menggunakan Docker. Untuk menjalankan API Inferensi dan MLflow Server secara bersamaan:

1. Pastikan Docker Desktop sudah berjalan di sistem Anda.
2. Buka terminal di direktori utama proyek.
3. Jalankan perintah orkestrasi berikut:
   ```bash
   docker compose up -d --build
   ```
4. Akses layanan melalui browser:
    - FastAPI (Swagger UI): http://localhost:8000/docs
    - MLflow Tracking Server: http://localhost:5000

5. Untuk mematikan seluruh layanan, gunakan perintah: 
    ```bash
    docker compose down
    ```

## Skalabilitas dan Kinerja Tinggi (Horizontal Scaling)

Sistem ini dirancang agar tahan banting (*fault-tolerant*) dan mampu menangani lonjakan lalu lintas (*traffic*) pengguna menggunakan fitur kloning kontainer dari Docker Compose. Secara bawaan, layanan inferensi API berjalan dengan **3 replika** secara paralel.

### Cara Mengakses Endpoint API yang Diskalakan

Sistem memetakan lalu lintas jaringan menggunakan rentang *port* (contoh: `8000-8002:8000`). Docker secara cerdas mendistribusikan masing-masing replika ke pintu masuk (*port*) yang berbeda di sistem operasi Anda. 

Anda dapat mengakses dokumentasi interaktif (Swagger UI) dari setiap replika secara terpisah melalui tautan berikut:
* **Replika 1:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Replika 2:** [http://localhost:8001/docs](http://localhost:8001/docs)
* **Replika 3:** [http://localhost:8002/docs](http://localhost:8002/docs)

### Cara Menambah/Mengurangi Replika Secara Dinamis

Anda dapat menskalakan (*scale up* atau *scale down*) jumlah kontainer API secara instan **tanpa harus mematikan server** yang sedang berjalan. Gunakan *flag* `--scale` pada terminal Anda:

```bash
# Contoh: Menskalakan layanan API menjadi 5 kontainer
docker compose up -d --scale api-service=5
```

**Catatan Penting:** Jika Anda ingin melakukan scale up lebih dari jumlah rentang port yang telah dideklarasikan di docker-compose.yaml (misalnya lebih dari 3), pastikan Anda juga memperlebar rentang port-nya terlebih dahulu (contoh: ubah menjadi "8000-8005:8000") agar kontainer baru tidak berebut port yang sama.

## Observabilitas & Pemantauan

Sistem ini dilengkapi dengan infrastruktur pemantauan (*monitoring*) tingkat produksi untuk melacak kesehatan API dan mendeteksi anomali model (*Data Drift*) secara *real-time*.

Metrik dikumpulkan secara otomatis menggunakan **Prometheus** dan divisualisasikan melalui **Grafana**. Anda dapat mengakses dasbor pemantauan melalui tautan berikut:

* **Prometheus (Target & Metrik Mentah):** [http://localhost:9090](http://localhost:9090)
* **Grafana (Dasbor Visualisasi):** [http://localhost:3000](http://localhost:3000)
  * *Default Username:* `admin`
  * *Default Password:* `admin`

*(Catatan: Grafana akan meminta Anda mengubah kata sandi pada saat login pertama kali, Anda dapat memilih "Skip" untuk melewati tahap tersebut).*