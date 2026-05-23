"""
export_model.py
Skrip untuk melatih 5 model Decision Tree sesuai dengan
Tahap_5_Comprehensive_Data_Mining.ipynb, lalu menyimpan ke .pkl
agar bisa dipakai oleh dashboard.

Eksperimen:
  1. BoW
  2. N-gram (Bigram)
  3. TF-IDF
  4. Word2Vec (Non-Contextual)
  5. BERT (Contextual - simulasi transfer learning)
"""
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from gensim.models import Word2Vec

# --- 1. Load Dataset ---
print("[1/6] Memuat dataset...")
df_train = pd.read_csv('../train.csv')
df_test = pd.read_csv('../test.csv')

# 13 label pertanian
agri_labels = ['grain', 'wheat', 'corn', 'sugar', 'oilseed', 'coffee', 'veg-oil', 'livestock', 'soybean', 'cocoa', 'carcass', 'cotton', 'barley']

X_train_raw = df_train['text'].astype(str).tolist()
y_train = df_train[agri_labels]

print(f"Menggunakan {len(agri_labels)} label pertanian: {agri_labels}")
print(f"Jumlah data train: {len(X_train_raw)}")

# --- 2. Data Augmentasi (Back Translation Simulation) ---
print("[2/6] Menjalankan data augmentasi...")

def back_translation_sim(texts, option='opsi_1'):
    augmented = []
    for text in texts:
        words = text.split()
        if option == 'opsi_1' and len(words) > 5:
            words[2], words[3] = words[3], words[2]
        elif option == 'opsi_3' and len(words) > 6:
            words[0], words[1] = words[1], words[0]
        augmented.append(' '.join(words))
    return augmented

minority_mask = (y_train['oilseed'] == 1) | (y_train['sugar'] == 1)
X_minority = np.array(X_train_raw)[minority_mask].tolist()
y_minority = y_train[minority_mask]

X_aug_1 = back_translation_sim(X_minority, option='opsi_1')
X_aug_3 = back_translation_sim(X_minority, option='opsi_3')

X_train_extended = X_train_raw + X_aug_1 + X_aug_3
y_train_extended = pd.concat([y_train, y_minority, y_minority], axis=0).reset_index(drop=True)

print(f"Augmentasi selesai. Data: {len(X_train_raw)} -> {len(X_train_extended)}")

# --- 3. Tokenisasi untuk Word2Vec/BERT ---
def clean_and_tokenize(text):
    return re.sub(r'[^a-zA-Z\s]', '', text.lower()).split()

X_train_tokens = [clean_and_tokenize(t) for t in X_train_extended]

# --- 4. Ekstraksi Fitur ---
print("[3/6] Membuat representasi fitur...")

# Eks-1: BoW
bow_vec = CountVectorizer(max_features=1000, stop_words='english')
X_train_bow = bow_vec.fit_transform(X_train_extended).toarray()

# Eks-2: N-gram
ngram_vec = CountVectorizer(ngram_range=(2, 2), max_features=1000, stop_words='english')
X_train_ngram = ngram_vec.fit_transform(X_train_extended).toarray()

# Eks-3: TF-IDF
tfidf_vec = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = tfidf_vec.fit_transform(X_train_extended).toarray()

# Eks-4: Word2Vec
w2v_model = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5, min_count=1, seed=42)

def get_w2v_avg(tokens):
    vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(100)

X_train_w2v = np.array([get_w2v_avg(t) for t in X_train_tokens])

# Eks-5: BERT (simulasi transfer learning)
X_train_bert = X_train_w2v * 1.05

# --- 5. Training Model (sesuai hyperparameter Comprehensive notebook) ---
print("[4/6] Melatih 5 model...")

def make_model():
    return MultiOutputClassifier(
        DecisionTreeClassifier(random_state=42, max_depth=7, min_samples_leaf=6, class_weight='balanced')
    )

# Eks-1: BoW
dt_bow = make_model()
dt_bow.fit(X_train_bow, y_train_extended)
print("  Model BoW selesai.")

# Eks-2: N-gram
dt_ngram = make_model()
dt_ngram.fit(X_train_ngram, y_train_extended)
print("  Model N-gram selesai.")

# Eks-3: TF-IDF
dt_tfidf = make_model()
dt_tfidf.fit(X_train_tfidf, y_train_extended)
print("  Model TF-IDF selesai.")

# Eks-4: Word2Vec
dt_w2v = make_model()
dt_w2v.fit(X_train_w2v, y_train_extended)
print("  Model Word2Vec selesai.")

# Eks-5: BERT
dt_bert = make_model()
dt_bert.fit(X_train_bert, y_train_extended)
print("  Model BERT selesai.")

# --- 6. Simpan semua model ---
print("[5/6] Menyimpan model ke file .pkl...")

with open('model_bow.pkl', 'wb') as f: pickle.dump(dt_bow, f)
with open('vectorizer_bow.pkl', 'wb') as f: pickle.dump(bow_vec, f)

with open('model_ngram.pkl', 'wb') as f: pickle.dump(dt_ngram, f)
with open('vectorizer_ngram.pkl', 'wb') as f: pickle.dump(ngram_vec, f)

with open('model_tfidf.pkl', 'wb') as f: pickle.dump(dt_tfidf, f)
with open('vectorizer_tfidf.pkl', 'wb') as f: pickle.dump(tfidf_vec, f)

with open('model_w2v.pkl', 'wb') as f: pickle.dump(dt_w2v, f)
with open('w2v_model.pkl', 'wb') as f: pickle.dump(w2v_model, f)

with open('model_bert.pkl', 'wb') as f: pickle.dump(dt_bert, f)
# BERT reuse w2v_model dengan faktor 1.05

with open('labels.pkl', 'wb') as f: pickle.dump(agri_labels, f)

print("[6/6] Selesai! Semua 5 model berhasil disimpan.")
print("File: model_bow.pkl, model_ngram.pkl, model_tfidf.pkl, model_w2v.pkl, model_bert.pkl")
