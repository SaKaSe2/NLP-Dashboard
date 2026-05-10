import streamlit as st
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="NLP Text Classification - Agriculture",
    page_icon="📰",
    layout="wide"
)

# Inisialisasi NLTK
@st.cache_resource(show_spinner="Mengunduh data NLTK...")
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_data()

# Pra-pemrosesan teks
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(cleaned_tokens)

# Memuat model berdasarkan pilihan
@st.cache_resource(show_spinner="Memuat model dari file...")
def load_model(model_type):
    if model_type == "Bag of Words (BoW)":
        model_file = 'model_bow.pkl'
        vec_file = 'vectorizer_bow.pkl'
    elif model_type == "N-Gram (Bigram)":
        model_file = 'model_ngram.pkl'
        vec_file = 'vectorizer_ngram.pkl'
    else: # Default TF-IDF
        model_file = 'model_tfidf.pkl'
        vec_file = 'vectorizer_tfidf.pkl'

    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(vec_file, 'rb') as f:
        vectorizer = pickle.load(f)
    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)
        
    return model, vectorizer, labels

# --- TAMPILAN DASHBOARD ---
st.title("Klasifikasi Artikel Berita Pertanian (NLP)")
st.write("Aplikasi ini menggunakan model **Decision Tree** untuk memprediksi klasifikasi multi-label artikel ke dalam **41 kategori pertanian** berdasarkan dataset Reuters-21578.")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Uji Prediksi Teks")
    
    # --- PILIHAN MODEL ---
    selected_model = st.radio(
        "Pilih Ekstraksi Fitur (Model) yang ingin digunakan:",
        ("Bag of Words (BoW)", "N-Gram (Bigram)", "TF-IDF"),
        index=2, # Default TF-IDF
        horizontal=True
    )
    
    dt_model, text_vectorizer, agri_labels = load_model(selected_model)

    st.caption("Ketik atau salin teks berita pertanian berbahasa Inggris di sini:")

    # Contoh teks pertanian yang bisa langsung dipakai
    sample_texts = {
        "-- Pilih contoh teks --": "",
        "Berita Gula (Sugar)": "EC beet sugar estimate for 1987/88 is unchanged at 12.63 mln tonnes white equivalent. Sugar production in Europe remains stable.",
        "Berita Gandum (Wheat)": "The wheat crop in the United States is expected to reach record levels this season. Farmers reported strong yields across the midwest wheat belt.",
        "Berita Jagung (Corn)": "Canada rules that U.S. corn imports are injuring Canadian farmers and upholds the anti-dumping duty on American corn shipments.",
        "Berita Biji-bijian (Grain)": "Japan should increase foreign access to its farm products market. Japan now produces only 30 pct of its annual grain needs, down from 61 pct some 20 years ago.",
    }
    selected_sample = st.selectbox("Atau pilih contoh teks:", list(sample_texts.keys()))

    if sample_texts[selected_sample]:
        default_text = sample_texts[selected_sample]
    else:
        default_text = ""

    user_input = st.text_area("Input teks:", value=default_text, height=150,
                              placeholder="Contoh: The agricultural department reported a significant increase in wheat and corn production this year...")

    if st.button("Analisis Teks", type="primary"):
        if user_input.strip():
            with st.spinner(f"Memproses menggunakan model {selected_model}..."):
                # 1. Cleaning
                cleaned_input = clean_text(user_input)
                # 2. Vectorization
                input_vec = text_vectorizer.transform([cleaned_input])
                # 3. Predict
                prediction = dt_model.predict(input_vec)[0]

                # 4. Ambil label yang bernilai 1 (Positif)
                predicted_labels = [agri_labels[i] for i, val in enumerate(prediction) if val == 1]

            st.success("Analisis Selesai!")

            if predicted_labels:
                st.write("### Kategori pertanian yang terdeteksi:")
                cols = st.columns(min(len(predicted_labels), 5))
                for idx, label in enumerate(predicted_labels):
                    with cols[idx % 5]:
                        st.info(f"**{label}**")
            else:
                st.warning("Tidak ada kategori pertanian yang terdeteksi. Teks ini kemungkinan bukan berita sektor pertanian, atau tidak mengandung kata kunci komoditas yang cukup kuat.")

            with st.expander("Lihat Teks yang Telah Dibersihkan (Pre-processed)"):
                st.write(cleaned_input)
        else:
            st.error("Harap masukkan teks terlebih dahulu!")

with col2:
    st.subheader("Cara Kerja Model (Visualisasi)")
    st.write("Visualisasi pohon keputusan di bawah ini adalah versi sederhana yang difokuskan pada kategori mayoritas (Grain) untuk memperlihatkan bagaimana model mengambil keputusan secara berjenjang.")
    if os.path.exists("decision_tree_vis.png"):
        st.image("decision_tree_vis.png", use_container_width=True)
    else:
        st.info("Gambar pohon keputusan tidak ditemukan.")

st.markdown("---")
st.subheader("Evaluasi dan Performa Keseluruhan")
tab_eval1, tab_eval2, tab_eval3 = st.tabs(["Perbandingan Metrik", "Confusion Matrix", "Feature Importance"])

with tab_eval1:
    if os.path.exists("eval_comparison.png"):
        st.image("eval_comparison.png", caption="Perbandingan Metrik Evaluasi untuk ke-3 Model", use_container_width=True)
with tab_eval2:
    col_cm1, col_cm2, col_cm3 = st.columns(3)
    with col_cm1:
        if os.path.exists("cm_bow.png"): st.image("cm_bow.png", caption="CM BoW", use_container_width=True)
    with col_cm2:
        if os.path.exists("cm_ngram.png"): st.image("cm_ngram.png", caption="CM N-Gram", use_container_width=True)
    with col_cm3:
        if os.path.exists("cm_tfidf.png"): st.image("cm_tfidf.png", caption="CM TF-IDF", use_container_width=True)

with tab_eval3:
    col_fi1, col_fi2, col_fi3 = st.columns(3)
    with col_fi1:
        if os.path.exists("fi_bow.png"): st.image("fi_bow.png", caption="FI BoW", use_container_width=True)
    with col_fi2:
        if os.path.exists("fi_ngram.png"): st.image("fi_ngram.png", caption="FI N-Gram", use_container_width=True)
    with col_fi3:
        if os.path.exists("fi_tfidf.png"): st.image("fi_tfidf.png", caption="FI TF-IDF", use_container_width=True)

# Daftar kategori yang didukung
with st.expander("Lihat 41 Kategori Pertanian yang Didukung"):
    cols_label = st.columns(5)
    for idx, label in enumerate(agri_labels):
        with cols_label[idx % 5]:
            st.write(f"- {label}")
