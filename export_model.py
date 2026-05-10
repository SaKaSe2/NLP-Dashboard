"""
export_model.py
Skrip untuk melatih model Decision Tree + TF-IDF menggunakan seluruh label
dari dataset train.csv, lalu menyimpan hasilnya ke file .pkl agar bisa
langsung dipakai oleh dashboard tanpa perlu re-training.
"""
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("[1/4] Memuat dataset...")
df = pd.read_csv('train.csv')

# Ambil seluruh label dari dataset (semua kolom kecuali 'text')
all_labels = [col for col in df.columns if col != 'text']
print(f"Total label ditemukan: {len(all_labels)}")

# Preprocessing
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

print("[2/4] Membersihkan teks...")
X_raw = df['text'].apply(clean_text)
y = df[all_labels].values

# TF-IDF Vectorization
print("[3/4] Melatih model TF-IDF + Decision Tree...")
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(X_raw)

dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_tfidf, y)

# Simpan ke file .pkl
print("[4/4] Menyimpan model ke file .pkl...")
with open('model_tfidf.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('labels.pkl', 'wb') as f:
    pickle.dump(all_labels, f)

print("Selesai! File yang dihasilkan:")
print("  - model_tfidf.pkl (Model Decision Tree)")
print("  - vectorizer.pkl  (TF-IDF Vectorizer)")
print("  - labels.pkl      (Daftar semua label)")
