import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

# Eks-1: BoW
bow = CountVectorizer(max_features=1000)
X_train_bow = bow.fit_transform(X_train)
dt_bow = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_bow.fit(X_train_bow, y_train)
rules_bow = export_text(dt_bow, feature_names=list(bow.get_feature_names_out()), max_depth=5)

# Eks-2: N-gram (Bigram)
ngram = CountVectorizer(ngram_range=(2, 2), max_features=1000)
X_train_ngram = ngram.fit_transform(X_train)
dt_ngram = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_ngram.fit(X_train_ngram, y_train)
rules_ngram = export_text(dt_ngram, feature_names=list(ngram.get_feature_names_out()), max_depth=5)

# Eks-3: TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
dt_tfidf = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_tfidf.fit(X_train_tfidf, y_train)
rules_tfidf = export_text(dt_tfidf, feature_names=list(tfidf.get_feature_names_out()), max_depth=5)

# Simpan ke file
with open('decision_tree_rules.txt', 'w', encoding='utf-8') as f:
    # Eks-1
    f.write("=" * 70 + "\n")
    f.write("EKSPERIMEN 1: POHON KEPUTUSAN (DT + Bag of Words)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Kedalaman Pohon  : {dt_bow.get_depth()}\n")
    f.write(f"Jumlah Daun      : {dt_bow.get_n_leaves()}\n")
    f.write(f"Jumlah Fitur     : {dt_bow.n_features_in_}\n")
    f.write("-" * 70 + "\n")
    f.write(rules_bow)
    f.write("\n\n")

    # Eks-2
    f.write("=" * 70 + "\n")
    f.write("EKSPERIMEN 2: POHON KEPUTUSAN (DT + N-gram / Bigram)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Kedalaman Pohon  : {dt_ngram.get_depth()}\n")
    f.write(f"Jumlah Daun      : {dt_ngram.get_n_leaves()}\n")
    f.write(f"Jumlah Fitur     : {dt_ngram.n_features_in_}\n")
    f.write("-" * 70 + "\n")
    f.write(rules_ngram)
    f.write("\n\n")

    # Eks-3
    f.write("=" * 70 + "\n")
    f.write("EKSPERIMEN 3: POHON KEPUTUSAN (DT + TF-IDF)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Kedalaman Pohon  : {dt_tfidf.get_depth()}\n")
    f.write(f"Jumlah Daun      : {dt_tfidf.get_n_leaves()}\n")
    f.write(f"Jumlah Fitur     : {dt_tfidf.n_features_in_}\n")
    f.write("-" * 70 + "\n")
    f.write(rules_tfidf)

print("Saved: decision_tree_rules.txt")
print(f"\nEks-1 (BoW)    : depth={dt_bow.get_depth()}, leaves={dt_bow.get_n_leaves()}")
print(f"Eks-2 (N-gram) : depth={dt_ngram.get_depth()}, leaves={dt_ngram.get_n_leaves()}")
print(f"Eks-3 (TF-IDF) : depth={dt_tfidf.get_depth()}, leaves={dt_tfidf.get_n_leaves()}")
