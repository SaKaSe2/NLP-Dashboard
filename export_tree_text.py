import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

df = pd.read_csv('train.csv')

agri_labels = [
    'barley', 'carcass', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil', 'coffee', 
    'copra-cake', 'corn', 'cotton', 'cotton-oil', 'grain', 'groundnut', 'groundnut-oil', 
    'hog', 'l-cattle', 'lin-oil', 'livestock', 'meal-feed', 'oat', 'oilseed', 'orange', 
    'palm-oil', 'palmkernel', 'potato', 'rape-oil', 'rapeseed', 'rice', 'rubber', 'rye', 
    'sorghum', 'soy-meal', 'soy-oil', 'soybean', 'sugar', 'sun-meal', 'sun-oil', 'sunseed', 
    'tea', 'veg-oil', 'wheat'
]

df_target = df[['text'] + agri_labels].copy()

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

df_target['cleaned_text'] = df_target['text'].apply(clean_text)

X = df_target['cleaned_text']
y = df_target[agri_labels].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)

# Decision Tree dengan kedalaman terbatas agar aturan teks tidak terlalu panjang
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train_tfidf, y_train)

# Export aturan pohon keputusan sebagai teks
tree_rules = export_text(dt, feature_names=list(tfidf.get_feature_names_out()), max_depth=5)

# Simpan ke file teks
with open('decision_tree_rules.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("POHON KEPUTUSAN (DECISION TREE) - ATURAN KLASIFIKASI\n")
    f.write("=" * 70 + "\n")
    f.write(f"Kedalaman Pohon  : {dt.get_depth()}\n")
    f.write(f"Jumlah Daun      : {dt.get_n_leaves()}\n")
    f.write(f"Jumlah Fitur     : {dt.n_features_in_}\n")
    f.write(f"Jumlah Label     : {len(agri_labels)} kategori pertanian\n")
    f.write("=" * 70 + "\n\n")
    f.write("Cara Membaca:\n")
    f.write("- Setiap baris '|--- fitur <= nilai' adalah kondisi percabangan\n")
    f.write("- 'class:' di ujung daun adalah prediksi label untuk artikel\n")
    f.write("- Contoh: jika kata 'wheat' memiliki skor TF-IDF <= 0.05,\n")
    f.write("  maka lanjut ke cabang kiri, dst.\n\n")
    f.write(tree_rules)

print("Decision tree rules saved to decision_tree_rules.txt")
print(f"\nInfo Pohon:")
print(f"  Kedalaman : {dt.get_depth()}")
print(f"  Jumlah Daun: {dt.get_n_leaves()}")
print(f"\nPreview aturan (50 baris pertama):")
for line in tree_rules.split('\n')[:50]:
    print(line)
