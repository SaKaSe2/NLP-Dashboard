# NLP Multi-Label Text Classification Dashboard

Dashboard berbasis web interaktif yang menggunakan **Streamlit** dan **Python** untuk melakukan *multi-label text classification* pada artikel/berita ke dalam **90 kategori** (berdasarkan dataset Reuters) menggunakan algoritma **Decision Tree** dengan ekstraksi fitur **TF-IDF**.

## Fitur Utama
1. **Prediksi Teks Real-Time**: Ketik atau salin teks berita, dan model akan langsung menampilkan kategori yang relevan.
2. **Model Siap Pakai (.pkl)**: Model yang sudah dilatih disimpan ke file `.pkl` sehingga aplikasi terbuka secara instan tanpa perlu melatih ulang.
3. **Visualisasi Pohon Keputusan**: Menyertakan ilustrasi representatif dari algoritma Decision Tree.
4. **Evaluasi Model Lengkap**: Menampilkan metrik perbandingan 3 eksperimen, Confusion Matrix, dan Feature Importance.

## Persyaratan Instalasi

1. **Clone Repository ini**
   ```bash
   git clone https://github.com/SaKaSe2/NLP-Dashboard.git
   cd NLP-Dashboard
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

4. **Export Model (Wajib dijalankan sekali sebelum membuka dashboard)**
   ```bash
   python export_model.py
   ```
   Perintah ini akan melatih model dan menghasilkan file `model_tfidf.pkl`, `vectorizer.pkl`, dan `labels.pkl`.

## Cara Menjalankan Dashboard
Setelah model berhasil di-export, jalankan:
```bash
streamlit run dashboard.py
```
Browser akan otomatis terbuka di `http://localhost:8501`.

## Cara Menjalankan Ulang Proses Pelatihan (Data Mining)
Jika Anda ingin melatih ulang model dan menghasilkan laporan evaluasi `.docx` beserta pembaruan gambar grafik:
```bash
python generate_final_report.py
```

---
*Proyek ini dikembangkan sebagai bagian dari Tugas Data Mining Tahap 5 - Klasifikasi Teks NLP.*
