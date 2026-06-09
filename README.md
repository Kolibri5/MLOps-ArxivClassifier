# MLOps ArXiv Paper Classifier 🚀

Proyek ini adalah implementasi *End-to-End Machine Learning Operations* (MLOps) untuk mengklasifikasikan abstrak jurnal ilmiah (ArXiv) ke dalam kategori tertentu. Sistem ini dilengkapi dengan *Continuous Training* (CT) otomatis yang dipicu oleh pemantauan *Data Drift* secara *real-time*.

## 🏗️ Arsitektur Sistem
* **Model Registry & Tracking:** MLflow
* **Inference API:** FastAPI (dengan *Endpoint* API Gateway untuk *Webhook*)
* **Monitoring & Metrics:** Prometheus
* **Visualization & Alerting:** Grafana
* **Continuous Integration / Continuous Training (CI/CD):** GitHub Actions
* **Containerization:** Docker & Docker Compose

### Alur Otomatisasi Continuous Training (CT)
1. User mengirimkan permintaan prediksi ke **FastAPI**.
2. FastAPI mencatat jumlah dan kategori prediksi ke dalam metrik **Prometheus**.
3. **Grafana** memantau metrik tersebut. Jika terjadi lonjakan anomali (*Data Drift*), Grafana akan membunyikan alarm (*Alert*) dan mengirimkan *Webhook* ke FastAPI.
4. *Endpoint* *Forwarder* di **FastAPI** menerjemahkan format *Webhook* Grafana dan meneruskannya ke **GitHub API** secara aman menggunakan *Personal Access Token* (PAT).
5. **GitHub Actions** mendeteksi panggilan tersebut dan secara otomatis menjalankan *pipeline* pelatihan ulang model (*Retrain Model*).

---

## ⚙️ Persiapan & Instalasi

### 1. Prasyarat
* Docker Desktop terinstal.
* Git terinstal.
* Akun GitHub dengan **Personal Access Token (PAT)** (Classic) yang memiliki izin *scope* `repo` dan `workflow`.

### 2. Kloning Repositori
```bash
git clone https://github.com/Kolibri5/mlops-arxivclassifier.git

cd mlops-arxivclassifier
```

### 3. Konfigurasi Environment Variables (Sangat Penting)
Demi keamanan, token GitHub tidak disimpan di dalam kode. Anda harus membuat brankas lokal menggunakan file `.env`.

Salin file *template* yang telah disediakan:
```bash
cp .env.example .env
```

Buka file `.env` di *Code Editor* Anda dan isi dengan token milik Anda:
```text
# Masukkan PAT GitHub Anda di bawah ini
GITHUB_PAT=ghp_TokenAsliAndaMasukkanDisini
GITHUB_DISPATCH_URL=[https://api.github.com/repos/USERNAME_ANDA/NAMA_REPO_ANDA/dispatches](https://api.github.com/repos/USERNAME_ANDA/NAMA_REPO_ANDA/dispatches)
MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

### 4. Menjalankan Sistem
Jalankan seluruh kontainer menggunakan Docker Compose:
```bash
docker compose up -d --build
```

---

## 📊 Akses Layanan

Setelah Docker berjalan, Anda dapat mengakses layanan-layanan berikut di *browser*:

| Layanan | URL Lokal | Keterangan |
|---------|-----------|------------|
| **FastAPI Swagger UI** | `http://localhost:8000/docs` | Dokumentasi dan *testing* API inferensi |
| **MLflow UI** | `http://localhost:5000` | Melihat *tracking* eksperimen dan model terdaftar |
| **Grafana** | `http://localhost:3000` | Dasbor pemantauan (Default login: `admin` / `admin`) |
| **Prometheus** | `http://localhost:9090` | Pengecekan *raw metrics* |

---

## 🔔 Konfigurasi Alert di Grafana

Karena dasbor dan *alert* memerlukan konfigurasi pengguna, ikuti langkah ini untuk menghubungkan Grafana dengan GitHub Actions:

1. Masuk ke Grafana (`http://localhost:3000`).
2. Buat **Contact Point** baru (Tipe: Webhook).
3. Masukkan URL: `http://api-service:8000/webhook/grafana-to-github` (Kosongkan bagian *username/password* karena token sudah ditangani oleh FastAPI secara internal melalui `.env`).
4. Buat **Alert Rule** berdasarkan metrik `model_predictions_total` dari Prometheus.
5. Arahkan **Notification Policies** ke *Contact Point* yang baru saja dibuat.

Kini, setiap kali lonjakan data terjadi, Grafana akan memicu proses pelatihan ulang (*retraining*) di GitHub Actions Anda secara otomatis!