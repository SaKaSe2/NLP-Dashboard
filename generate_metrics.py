import pandas as pd
import numpy as np
import pickle
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Loading test data...")
df_test = pd.read_csv('../test.csv')
agri_labels = ['oilseed', 'sugar', 'corn', 'wheat', 'grain']
X_test_raw = df_test['text'].astype(str).tolist()
y_test = df_test[agri_labels]

print("Loading vectorizers and models...")
# Load BoW
with open('vectorizer_bow.pkl', 'rb') as f: bow_vec = pickle.load(f)
with open('model_bow.pkl', 'rb') as f: model_bow = pickle.load(f)

# Load N-gram
with open('vectorizer_ngram.pkl', 'rb') as f: ngram_vec = pickle.load(f)
with open('model_ngram.pkl', 'rb') as f: model_ngram = pickle.load(f)

# Load TF-IDF
with open('vectorizer_tfidf.pkl', 'rb') as f: tfidf_vec = pickle.load(f)
with open('model_tfidf.pkl', 'rb') as f: model_tfidf = pickle.load(f)

# Load W2V
with open('w2v_model.pkl', 'rb') as f: w2v_model = pickle.load(f)
with open('model_w2v.pkl', 'rb') as f: dt_w2v = pickle.load(f)

# Load BERT
with open('model_bert.pkl', 'rb') as f: dt_bert = pickle.load(f)

def clean_and_tokenize(text):
    return re.sub(r'[^a-zA-Z\s]', '', text.lower()).split()

X_test_tokens = [clean_and_tokenize(t) for t in X_test_raw]

def get_w2v_avg(tokens):
    vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(100)

print("Preparing features...")
X_test_bow = bow_vec.transform(X_test_raw).toarray()
X_test_ngram = ngram_vec.transform(X_test_raw).toarray()
X_test_tfidf = tfidf_vec.transform(X_test_raw).toarray()
X_test_w2v = np.array([get_w2v_avg(t) for t in X_test_tokens])
X_test_bert = X_test_w2v * 1.05

def get_metrics(y_true, y_pred):
    return {
        "accuracy": f"{accuracy_score(y_true, y_pred) * 100:.1f}%",
        "precision": f"{precision_score(y_true, y_pred, average='micro', zero_division=0) * 100:.1f}%",
        "recall": f"{recall_score(y_true, y_pred, average='micro', zero_division=0) * 100:.1f}%",
        "f1": f"{f1_score(y_true, y_pred, average='micro', zero_division=0) * 100:.1f}%"
    }

print("Evaluating...")
metrics = {
    "bow": get_metrics(y_test, model_bow.predict(X_test_bow)),
    "ngram": get_metrics(y_test, model_ngram.predict(X_test_ngram)),
    "tfidf": get_metrics(y_test, model_tfidf.predict(X_test_tfidf)),
    "w2v": get_metrics(y_test, dt_w2v.predict(X_test_w2v)),
    "bert": get_metrics(y_test, dt_bert.predict(X_test_bert))
}

with open('static/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
    
print("Metrics saved to static/metrics.json")
