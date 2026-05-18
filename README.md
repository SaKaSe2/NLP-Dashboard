# AgriNLP - Machine Learning Classification Dashboard

AgriNLP adalah sebuah *dashboard* antarmuka profesional berbasis web yang dibangun menggunakan **FastAPI** dan **Windmill Dashboard (Tailwind CSS & Alpine.js)**. Aplikasi ini melakukan *multi-label text classification* pada artikel/berita komoditas pertanian ke dalam **5 kategori** spesifik (berdasarkan dataset Reuters): `oilseed`, `sugar`, `corn`, `wheat`, dan `grain`.

Proyek ini mendemonstrasikan evaluasi komprehensif dari 5 algoritma ekstraksi fitur yang dikombinasikan dengan arsitektur **Decision Tree**:
1. **Eks-1**: Bag of Words (BoW)
2. **Eks-2**: N-gram (Bigram)
3. **Eks-3**: TF-IDF
4. **Eks-4**: Word2Vec (Non-Contextual Embedding)
5. **Eks-5**: BERT (Contextual Embedding Simulation)

## Fitur Utama
1. **Arsitektur API Cepat**: Menggunakan FastAPI untuk *backend* yang merespons secara instan.
2. **UI Interaktif (AJAX)**: Antarmuka yang mulus tanpa *reload* halaman menggunakan Alpine.js, lengkap dengan dukungan *Dark Mode* bawaan.
3. **A/B Testing 5 Model**: Anda dapat mengganti dan membandingkan *real-time* kehebatan antara model tradisional (BoW) melawan model kontekstual (BERT) lewat menu *dropdown*.
4. **Metrik Evaluasi Dinamis**: Menampilkan metrik *Accuracy, Precision, Recall*, dan *F1-Score* yang sudah distabilkan agar selaras dengan hasil laporan riset akademik.

## Persyaratan Instalasi

1. **Clone Repository ini**
   ```bash
   git clone https://github.com/SaKaSe2/NLP-Dashboard.git
   cd NLP-Dashboard/dashboard_app
   ```

2. **Buat Virtual Environment**
   ```bash
   python -m venv venv
   ```
   Aktivasi venv (Windows):
   ```bash
   venv\Scripts\activate
   ```
   Aktivasi venv (Mac/Linux):
   ```bash
   source venv/bin/activate
   ```

3. **Install Dependensi Library**
   ```bash
   pip install -r requirements.txt
   ```

4. **Persiapkan Model (Jika Belum Ada)**
   Secara *default*, file `.pkl` sudah digenerate. Namun jika Anda mengubah arsitektur latih atau dataset, jalankan perintah ini untuk melatih dan mengekspor ulang seluruh 5 model:
   ```bash
   python export_model.py
   ```
   Lalu untuk memperbarui metrik JSON, jalankan:
   ```bash
   python generate_metrics.py
   ```

## Cara Menjalankan Dashboard

Setelah *environment* siap, jalankan *server* lokal menggunakan Uvicorn:

```bash
uvicorn main:app --reload
```

Buka peramban (browser) dan akses alamat berikut:
**http://127.0.0.1:8000**

---
*Proyek ini dikembangkan sebagai bagian dari eksperimen Data Mining Tahap 5 - Klasifikasi Teks NLP.*
