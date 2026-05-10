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

from sklearn.feature_extraction.text import CountVectorizer

# 1. Bag of Words (BoW)
print("[3/4] Melatih model 1: Bag of Words (BoW)...")
bow_vec = CountVectorizer(max_features=1000)
X_bow = bow_vec.fit_transform(X_raw)
dt_bow = DecisionTreeClassifier(random_state=42)
dt_bow.fit(X_bow, y)

with open('model_bow.pkl', 'wb') as f: pickle.dump(dt_bow, f)
with open('vectorizer_bow.pkl', 'wb') as f: pickle.dump(bow_vec, f)

# 2. N-Gram (Bigram)
print("[4/4] Melatih model 2: N-Gram (Bigram)...")
ngram_vec = CountVectorizer(ngram_range=(2,2), max_features=1000)
X_ngram = ngram_vec.fit_transform(X_raw)
dt_ngram = DecisionTreeClassifier(random_state=42)
dt_ngram.fit(X_ngram, y)

with open('model_ngram.pkl', 'wb') as f: pickle.dump(dt_ngram, f)
with open('vectorizer_ngram.pkl', 'wb') as f: pickle.dump(ngram_vec, f)

# 3. TF-IDF
print("[5/4] Melatih model 3: TF-IDF...")
tfidf_vec = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vec.fit_transform(X_raw)
dt_tfidf = DecisionTreeClassifier(random_state=42)
dt_tfidf.fit(X_tfidf, y)

with open('model_tfidf.pkl', 'wb') as f: pickle.dump(dt_tfidf, f)
with open('vectorizer_tfidf.pkl', 'wb') as f: pickle.dump(tfidf_vec, f)

# Simpan label
with open('labels.pkl', 'wb') as f:
    pickle.dump(agri_labels, f)

print("Selesai! File yang dihasilkan untuk BoW, N-Gram, dan TF-IDF sudah disimpan.")
