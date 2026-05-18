"""
main.py
FastAPI backend untuk Dashboard NLP Classification.
Mendukung 5 eksperimen model sesuai Tahap_5_Comprehensive_Data_Mining.ipynb.
"""
import os
import re
import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="NLP Agriculture Classification Dashboard")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# --- Load semua model saat startup ---
models = {}
vectorizers = {}
w2v_model = None
agri_labels = []

@app.on_event("startup")
def load_all_models():
    """Load semua model dan vectorizer dari file .pkl."""
    global models, vectorizers, w2v_model, agri_labels

    with open("labels.pkl", "rb") as f:
        agri_labels = pickle.load(f)

    # Traditional models (BoW, N-gram, TF-IDF)
    for name, model_file, vec_file in [
        ("bow", "model_bow.pkl", "vectorizer_bow.pkl"),
        ("ngram", "model_ngram.pkl", "vectorizer_ngram.pkl"),
        ("tfidf", "model_tfidf.pkl", "vectorizer_tfidf.pkl"),
    ]:
        with open(model_file, "rb") as f:
            models[name] = pickle.load(f)
        with open(vec_file, "rb") as f:
            vectorizers[name] = pickle.load(f)

    # Word2Vec model (dipakai oleh w2v dan bert)
    with open("w2v_model.pkl", "rb") as f:
        w2v_model = pickle.load(f)

    # Embedding models (Word2Vec, BERT)
    with open("model_w2v.pkl", "rb") as f:
        models["w2v"] = pickle.load(f)
    with open("model_bert.pkl", "rb") as f:
        models["bert"] = pickle.load(f)

    # Simpan w2v_model ke vectorizers agar mudah diakses
    vectorizers["w2v"] = w2v_model
    vectorizers["bert"] = w2v_model

    print(f"Loaded {len(models)} models, labels: {agri_labels}")


# --- Helper functions ---
def clean_text(text: str) -> str:
    """Bersihkan teks input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_and_tokenize(text: str) -> list:
    """Tokenisasi untuk Word2Vec/BERT."""
    return re.sub(r"[^a-zA-Z\s]", "", text.lower()).split()

def get_w2v_avg(tokens: list, w2v) -> np.ndarray:
    """Hitung rata-rata vektor Word2Vec."""
    vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
    return np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(100)

def extract_decision_path(estimator, input_vec, vec, model_key):
    """Mengekstrak seluruh pohon keputusan dan menyorot jalur yang aktif."""
    tree_ = estimator.tree_
    node_indicator = estimator.decision_path(input_vec)
    active_nodes = set(node_indicator.indices)
    
    vocab = {}
    if model_key in ("bow", "ngram", "tfidf") and hasattr(vec, "vocabulary_"):
        vocab = {v: k for k, v in vec.vocabulary_.items()}

    nodes = []
    edges = []

    for i in range(tree_.node_count):
        if tree_.feature[i] != -2: # bukan leaf
            feature_idx = tree_.feature[i]
            feature_name = vocab.get(feature_idx, f"Dim_{feature_idx}")
            threshold = tree_.threshold[i]
            label = f"{feature_name} <= {threshold:.2f}"
            nodes.append(f"    N{i}[\"{label}\"]")
        else: # leaf
            val = tree_.value[i][0]
            predicted_class = int(np.argmax(val))
            prob = val[predicted_class] / np.sum(val)
            label = f"Class {predicted_class}\\n({prob*100:.0f}%)"
            nodes.append(f"    N{i}((\"{label}\"))")
            
        left_child = tree_.children_left[i]
        right_child = tree_.children_right[i]
        
        if left_child != -1:
            edges.append(f"    N{i} -->|True| N{left_child}")
        if right_child != -1:
            edges.append(f"    N{i} -->|False| N{right_child}")

    mermaid_str = "graph TD\n"
    mermaid_str += "\n".join(nodes) + "\n"
    mermaid_str += "\n".join(edges) + "\n"
    
    for node_id in active_nodes:
        mermaid_str += f"    style N{node_id} fill:#7e3af2,stroke:#fff,color:#fff\n"

    return mermaid_str

def predict_text(text: str, model_key: str) -> dict:
    """Prediksi teks menggunakan model yang dipilih."""
    model = models[model_key]
    cleaned = clean_text(text)

    if model_key in ("bow", "ngram", "tfidf"):
        # Traditional: gunakan vectorizer
        vec = vectorizers[model_key]
        input_vec = vec.transform([cleaned]).toarray()
    elif model_key == "w2v":
        # Word2Vec: rata-rata vektor
        tokens = clean_and_tokenize(text)
        avg = get_w2v_avg(tokens, vectorizers["w2v"])
        input_vec = avg.reshape(1, -1)
    else:
        # BERT: w2v * 1.05
        tokens = clean_and_tokenize(text)
        avg = get_w2v_avg(tokens, vectorizers["bert"]) * 1.05
        input_vec = avg.reshape(1, -1)

    prediction = model.predict(input_vec)[0]
    predicted_labels = [agri_labels[i] for i, val in enumerate(prediction) if val == 1]
    
    # Get probabilities for chart
    proba = model.predict_proba(input_vec)
    probabilities = {agri_labels[i]: round(p[0][1] * 100, 2) for i, p in enumerate(proba)}

    # Extract Decision Path
    # Pilih kategori yang terdeteksi, atau kategori dengan probabilitas tertinggi jika tidak ada
    target_idx = 0
    if 1 in prediction:
        target_idx = prediction.tolist().index(1)
    else:
        probs = [p[0][1] for p in proba]
        target_idx = np.argmax(probs)
        
    path_label = agri_labels[target_idx]
    
    # Hanya generate decision tree untuk model tradisional (Eks 1-3)
    decision_path = None
    if model_key in ("bow", "ngram", "tfidf"):
        decision_path = extract_decision_path(model.estimators_[target_idx], input_vec, vectorizers.get(model_key), model_key)

    return {
        "cleaned_text": cleaned,
        "predicted_labels": predicted_labels,
        "raw_prediction": prediction.tolist(),
        "probabilities": probabilities,
        "decision_path": decision_path,
        "path_label": path_label.upper(),
        "model_used": model_key,
    }


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render halaman utama dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "labels": agri_labels},
    )

@app.post("/predict")
async def predict(request: Request):
    """Endpoint prediksi via AJAX (JSON response)."""
    body = await request.json()
    text = body.get("text", "")
    model_key = body.get("model", "bert")

    if not text.strip():
        return JSONResponse({"error": "Teks tidak boleh kosong"}, status_code=400)

    if model_key not in models:
        return JSONResponse({"error": f"Model '{model_key}' tidak tersedia"}, status_code=400)

    result = predict_text(text, model_key)
    return JSONResponse(result)

@app.get("/api/labels")
async def get_labels():
    """Return daftar label yang didukung."""
    return JSONResponse({"labels": agri_labels})

@app.get("/api/models")
async def get_models():
    """Return daftar model yang tersedia."""
    return JSONResponse({
        "models": [
            {"key": "bow", "name": "Bag of Words (BoW)", "experiment": "Eks-1"},
            {"key": "ngram", "name": "N-Gram (Bigram)", "experiment": "Eks-2"},
            {"key": "tfidf", "name": "TF-IDF", "experiment": "Eks-3"},
            {"key": "w2v", "name": "Word2Vec (Non-Contextual)", "experiment": "Eks-4"},
            {"key": "bert", "name": "BERT (Contextual)", "experiment": "Eks-5"},
        ]
    })
