# 🌾 NLP Text Classification & Dashboard - Agriculture

Repositori ini berisi **Aplikasi Web Interaktif (Dashboard)** dan **Kode Sumber Data Mining** untuk memproses *multi-label text classification* pada artikel pertanian (41 kategori) menggunakan algoritma **Decision Tree** dengan ekstraksi fitur **TF-IDF**.

## 🚀 Fitur Utama
1. **Aplikasi Dashboard (Streamlit)**: Memungkinkan uji teks secara *real-time* untuk memprediksi kategori hasil pertanian beserta visualisasinya.
2. **Skrip Eksekusi Otomatis**: Menyediakan skrip `run_tahap5.py` dan `generate_final_report.py` untuk menjalankan seluruh eksperimen komparasi (BoW, N-Gram, TF-IDF).

## 🛠️ Persyaratan Instalasi
Pastikan Anda sudah menginstal Python di komputer Anda. Sangat disarankan menggunakan **Virtual Environment**.

1. **Clone Repository ini**
   ```bash
   git clone https://github.com/SaKaSe2/NLP-Dashboard.git
   cd NLP-Dashboard
   ```

2. **Buat Virtual Environment (Sangat disarankan)**
   ```bash
   python -m venv venv
   ```
   *Aktivasi venv (Windows):*
   ```bash
   venv\Scripts\activate
   ```
   *Aktivasi venv (Mac/Linux):*
   ```bash
   source venv/bin/activate
   ```

3. **Install Dependensi Library**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Cara Menjalankan

### 1. Menjalankan Dashboard Interaktif
Pastikan *virtual environment* Anda aktif, lalu jalankan perintah ini di terminal:
```bash
streamlit run dashboard.py
```
Browser akan otomatis terbuka di `http://localhost:8501`.

### 2. Menjalankan Ulang Proses Pelatihan (Data Mining)
Jika Anda ingin melatih ulang model dan menghasilkan laporan evaluasi `.docx` beserta pembaruan gambar matriks terbaru, Anda cukup menjalankan:
```bash
python generate_final_report.py
```
*(Catatan: pastikan file dataset `train.csv` berada di folder yang sama).*

---
*Proyek ini dikembangkan sebagai bagian dari Tugas Data Mining Tahap 5 - Klasifikasi Teks NLP.*
