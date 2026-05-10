import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure resources are available
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

def evaluate_model(X_train_feat, X_test_feat, name):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_feat, y_train)
    y_pred = dt.predict(X_test_feat)
    
    acc = accuracy_score(y_test, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(y_test, y_pred, average='micro', zero_division=0)
    
    print(f"| {name} | {acc:.4f} | {p_mac:.4f} | {r_mac:.4f} | {f1_mac:.4f} | {p_mic:.4f} | {r_mic:.4f} | {f1_mic:.4f} |")
    return dt, y_pred

print("| Eksperimen | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Precision (Micro) | Recall (Micro) | F1-Score (Micro) |")
print("|---|---|---|---|---|---|---|---|")

# Eks-1: BoW
bow = CountVectorizer(max_features=1000)
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)
dt_bow, y_pred_bow = evaluate_model(X_train_bow, X_test_bow, "Eks-1: DT + BoW")

# Eks-2: N-gram (Bigram)
ngram = CountVectorizer(ngram_range=(2, 2), max_features=1000)
X_train_ngram = ngram.fit_transform(X_train)
X_test_ngram = ngram.transform(X_test)
dt_ngram, y_pred_ngram = evaluate_model(X_train_ngram, X_test_ngram, "Eks-2: DT + N-gram")

# Eks-3: TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
dt_tfidf, y_pred_tfidf = evaluate_model(X_train_tfidf, X_test_tfidf, "Eks-3: DT + TF-IDF")

top_5_labels = np.argsort(y.sum(axis=0))[-5:]
top_5_names = [agri_labels[i] for i in top_5_labels]

mcm = multilabel_confusion_matrix(y_test, y_pred_tfidf)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, (label_idx, label_name) in enumerate(zip(top_5_labels, top_5_names)):
    sns.heatmap(mcm[label_idx], annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f'CM: {label_name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_tfidf.png')
print("\nConfusion matrix plot saved to confusion_matrix_tfidf.png")

dt_simple = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_simple.fit(X_train_tfidf, y_train)
plt.figure(figsize=(20, 10))
plot_tree(dt_simple, feature_names=tfidf.get_feature_names_out(), class_names=True, filled=True, rounded=True, fontsize=10)
plt.title('Pohon Keputusan (Decision Tree) - Max Depth 3 (Simplified for Visualization)')
plt.savefig('decision_tree_vis.png')
print("Decision tree plot saved to decision_tree_vis.png")
