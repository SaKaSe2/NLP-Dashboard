"""
export_model.py
Skrip untuk melatih model Decision Tree + TF-IDF menggunakan 41 label pertanian
dari dataset train.csv (sesuai Tahap 2: Target Data pada laporan),
lalu menyimpan hasilnya ke file .pkl agar bisa langsung dipakai oleh dashboard.
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

# 41 label pertanian (sesuai Tahap 2: Target Data pada laporan)
agri_labels = [
    'barley', 'carcass', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil', 'coffee', 
    'copra-cake', 'corn', 'cotton', 'cotton-oil', 'grain', 'groundnut', 'groundnut-oil', 
    'hog', 'l-cattle', 'lin-oil', 'livestock', 'meal-feed', 'oat', 'oilseed', 'orange', 
    'palm-oil', 'palmkernel', 'potato', 'rape-oil', 'rapeseed', 'rice', 'rubber', 'rye', 
    'sorghum', 'soy-meal', 'soy-oil', 'soybean', 'sugar', 'sun-meal', 'sun-oil', 'sunseed', 
    'tea', 'veg-oil', 'wheat'
]
print(f"Menggunakan {len(agri_labels)} label pertanian")

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
y = df[agri_labels].values

# TF-IDF Vectorization
print("[3/4] Melatih model TF-IDF + Decision Tree (max_depth=5)...")
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(X_raw)

# Model tanpa batasan kedalaman agar prediksi lebih akurat di dashboard
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_tfidf, y)

# Simpan ke file .pkl
print("[4/4] Menyimpan model ke file .pkl...")
with open('model_tfidf.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('labels.pkl', 'wb') as f:
    pickle.dump(agri_labels, f)

print("Selesai! File yang dihasilkan:")
print("  - model_tfidf.pkl (Model Decision Tree)")
print("  - vectorizer.pkl  (TF-IDF Vectorizer)")
print("  - labels.pkl      (41 label pertanian)")
