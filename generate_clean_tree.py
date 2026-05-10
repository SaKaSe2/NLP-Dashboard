import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os

print("Memuat data dan membersihkan teks...")
df = pd.read_csv('train.csv')

agri_labels = [
    'barley', 'carcass', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil', 'coffee', 
    'copra-cake', 'corn', 'cotton', 'cotton-oil', 'grain', 'groundnut', 'groundnut-oil', 
    'hog', 'l-cattle', 'lin-oil', 'livestock', 'meal-feed', 'oat', 'oilseed', 'orange', 
    'palm-oil', 'palmkernel', 'potato', 'rape-oil', 'rapeseed', 'rice', 'rubber', 'rye', 
    'sorghum', 'soy-meal', 'soy-oil', 'soybean', 'sugar', 'sun-meal', 'sun-oil', 'sunseed', 
    'tea', 'veg-oil', 'wheat'
]

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

X_raw = df['text'].apply(clean_text)

# Pilih kategori mayoritas saja agar tree-nya bersih
y_counts = df[agri_labels].sum(axis=0)
top_label = y_counts.idxmax()
print(f"Kategori terbanyak adalah '{top_label}'. Membangun model representatif untuk visualisasi...")

y = df[top_label].values

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(X_raw)

# Kedalaman max 3 agar terlihat sederhana dan rapi seperti ilustrasi
dt_clean = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_clean.fit(X_tfidf, y)

print("Menggambar plot...")
plt.figure(figsize=(16, 8))
plot_tree(dt_clean, feature_names=tfidf.get_feature_names_out(), class_names=[f'Not {top_label.capitalize()}', top_label.capitalize()],
          filled=True, rounded=True, fontsize=12, impurity=False, proportion=False)

plt.title(f'Ilustrasi Pohon Keputusan (Decision Tree)\nBerdasarkan Model Klasifikasi Kategori "{top_label.capitalize()}"', fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('decision_tree_vis.png', dpi=150, bbox_inches='tight')
print("Selesai! decision_tree_vis.png telah diperbarui menjadi lebih bersih.")
