--- MARKDOWN ---
# Tahap 3-4: Data Pre-processing & Transformation

## Business Objective & Original Data
Mini project ini berfokus pada klasifikasi artikel berita ke dalam kategori-kategori pertanian secara otomatis menggunakan teknik data mining. Dataset yang digunakan adalah Reuters-21578, yaitu kumpulan artikel berita dari kantor berita Reuters yang mencakup 90 label topik berbeda. Banyak diantaranya berkaitan langsung dengan sektor pertanian, seperti grain, wheat, corn, barley, rice, hingga livestock.

Tujuan dari project ini adalah membangun model klasifikasi yang mampu secara otomatis menentukan label pertanian yang sesuai dari teks sebuah artikel berita (multi-label text classification). Notebook ini secara spesifik menangani alur **Data Pre-processing & Transformation** seperti Data Cleaning, Reduction, Transformation, Normalization, hingga Feature Selection yang dijabarkan beserta visualisasinya.

--- CODE ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from wordcloud import WordCloud

# Download resource yang dibutuhkan NLTK untuk Cleaning
nltk.download('stopwords')
nltk.download('wordnet')

--- MARKDOWN ---
## 1. Load Data & Target Data Selection (Data Reduction)

**Tahap 2**: Dari 91 kolom (1 text + 90 label), kita melakukan **Data Reduction** dengan mendrop 49 label non-pertanian dan hanya me-retain (menggunakan) atribut `text` serta 41 label yang berkaitan dengan pertanian.

--- CODE ---
from IPython.display import display

# Load dataset
df = pd.read_csv('train.csv')

# Tampilkan info data awal (Before Reduction)
print(f"[*] BEFORE DATA REDUCTION:")
print(f"Dimensi data original : {df.shape}")
display(df.head())

# 41 Label berdasarkan ruang lingkup pertanian dataset Reuters
agri_labels = [
    'barley', 'carcass', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil', 'coffee', 
    'copra-cake', 'corn', 'cotton', 'cotton-oil', 'grain', 'groundnut', 'groundnut-oil', 
    'hog', 'l-cattle', 'lin-oil', 'livestock', 'meal-feed', 'oat', 'oilseed', 'orange', 
    'palm-oil', 'palmkernel', 'potato', 'rape-oil', 'rapeseed', 'rice', 'rubber', 'rye', 
    'sorghum', 'soy-meal', 'soy-oil', 'soybean', 'sugar', 'sun-meal', 'sun-oil', 'sunseed', 
    'tea', 'veg-oil', 'wheat'
]

# Ambil kolom text dan label pertanian
df_target = df[['text'] + agri_labels].copy()

print(f"\n[*] AFTER DATA REDUCTION:")
print(f"Dimensi data reduksi  : {df_target.shape} (Hanya Text & Label Pertanian)")
display(df_target.head())

--- MARKDOWN ---
## 2. Visualisasi Data Original
Mari kita visualisasikan distribusi imbalance masing-masing kategori label.

--- CODE ---
# Menghitung frekuensi kemunculan berita untuk setiap label
label_counts = df_target[agri_labels].sum().sort_values(ascending=False)

# 1. Plot Distribusi Kategori
plt.figure(figsize=(14, 10))
sns.barplot(x=label_counts.values, y=label_counts.index, palette='crest')
plt.title('Distribusi Data: Frekuensi Artikel berdasarkan Kategori Pertanian')
plt.xlabel('Jumlah Artikel Berita')
plt.ylabel('Kategori')
plt.show()

# 2. Plot Distribusi Label per Artikel (melihat multi-labeling)
label_per_article = df_target[agri_labels].sum(axis=1)
plt.figure(figsize=(8, 5))
sns.countplot(x=label_per_article, palette='mako')
plt.title('Distribusi Banyaknya Label per Artikel Berita')
plt.xlabel('Jumlah Kategori dalam 1 Artikel')
plt.ylabel('Jumlah Artikel')
plt.show()

--- MARKDOWN ---
## 3. Data Cleaning
Membersihkan teks dari elemen pengganggu dan melakukan text normalization.
Tahapan cleaning:
1. **Case Folding** (Lowercase)
2. **Hapus Tanda Baca, Link, Angka** (Karakter Non-Alfabet)
3. **Tokenisasi & Removes Stopwords** (Menghilangkan kata hubung bahasa inggris seperti the, is, and)
4. **Lemmatization** (Mengembalikan kata ke bentuk dasar, misal *horses* menjadi *horse*)

--- CODE ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove karakter khusus dan angka
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Hapus spasi ganda
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3 & 4. Tokenisasi, Hapus stopwords & Lemmatization
    tokens = text.split()
    # Tambahan: hapus huruf tunggal (len > 1) agar teks lebih mudah dibaca dan bebas dari sisa huruf singkatan 
    # (misal 'U.S.' tidak menyisakan huruf 'u' saja).
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    
    return " ".join(cleaned_tokens)

# Mengaplikasikan fungsi
df_target['cleaned_text'] = df_target['text'].apply(clean_text)

print("[*] Perbandingan 5 Data BEFORE dan AFTER Cleaning:")
# Tampilkan DataFrame dengan max_colwidth=None agar kalimat tidak terpotong (utuh)
pd.set_option('display.max_colwidth', None)
from IPython.display import display
display(df_target[['text', 'cleaned_text']].head(5))
pd.reset_option('display.max_colwidth')


--- MARKDOWN ---
## 4. Visualisasi Data Teks Bersih (WordCloud)
Untuk melihat kata kunci mayoritas yang muncul setelah `stopwords` dihilangkan.

--- CODE ---
all_texts = " ".join(df_target['cleaned_text'])

wordcloud = WordCloud(width=900, height=500, 
                      background_color='white', 
                      colormap='Dark2',
                      max_words=200).generate(all_texts)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud: Kata Teratas Pada Artikel Pertanian (Reuters)')
plt.show()

--- MARKDOWN ---
## 5. Data Integration
Karena dataset (Reuters) yang kita gunakan sudah berbentuk tabel tunggal utuh (`train.csv`) dan relevan secara lengkap yang mencakup atribut Teks dan 90 varian Label, maka proses **Data integration** (penggabungan multiple data source dan skema pencocokan id) secara struktural telah dilakukan dari awal. Jika terdapat uji validasi, misal berasal dataset eksternal (seperti test.csv) baru iterasi _concat/merge_ table diperlukan.

--- MARKDOWN ---
## 6.1 Data Transformation (Ekstraksi Fitur)
Mengubah teks bersih menjadi fitur matriks numerik agar algoritma machine learning bisa memahaminya. Kita mencontohkan tiga jenis representasi fitur dasar dari Scikit-Learn:
- **Bag of Words (BoW)**: Menghitung frekuensi kemunculan kata pada dokumen.
- **N-gram**: Mengekstrak kombinasi $n$ kata yang berurutan secara bersamaan (contoh ini menggunakan Bigram, yaitu fitur gabungan 2 kata berurutan).
- **TF-IDF Vectorizer**: Menghitung bobot kata berdasarkan *Term Frequency* dan *Inverse Document Frequency*. Matriks TF-IDF inilah yang akan diteruskan pada tahapan Machine Learning (Normalisasi & Binarisasi).


--- CODE ---
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from IPython.display import display

print("[*] 5 Data BEFORE Transformation (Teks Bersih):")
pd.set_option('display.max_colwidth', None)
display(df_target[['cleaned_text']].head(5))
pd.reset_option('display.max_colwidth')

# 1. Bag of Words (BoW)
bow = CountVectorizer(max_features=1000)
X_bow = bow.fit_transform(df_target['cleaned_text'])

print(f"\n[*] Dimensi Fitur BoW : {X_bow.shape}")
print("[*] 5 Data AFTER Transformation (Bag of Words) - Menampilkan kolom yang memiliki nilai:")
bow_feature_names = bow.get_feature_names_out()
df_bow = pd.DataFrame(X_bow[:5].toarray(), columns=bow_feature_names)
display(df_bow.loc[:, df_bow.sum() > 0]) # Hanya tampilkan kolom yang bernilai > 0 di 5 data ini

# 2. N-gram (Bigram)
ngram = CountVectorizer(ngram_range=(2, 2), max_features=1000)
X_ngram = ngram.fit_transform(df_target['cleaned_text'])

print(f"\n[*] Dimensi Fitur N-gram (Bigram) : {X_ngram.shape}")
print("[*] 5 Data AFTER Transformation (N-gram) - Menampilkan kolom yang memiliki nilai:")
ngram_feature_names = ngram.get_feature_names_out()
df_ngram = pd.DataFrame(X_ngram[:5].toarray(), columns=ngram_feature_names)
display(df_ngram.loc[:, df_ngram.sum() > 0]) # Hanya tampilkan kolom yang bernilai > 0 di 5 data ini

# 3. Feature Extraction / Transformation (TF-IDF)
tfidf = TfidfVectorizer(
    max_features=1000,   # Hanya ambil 1000 kata dengan skor TF-IDF tertinggi
    norm=None            # Dimatikan sementara untuk didemonstrasikan di tahap normalisasi selanjutnya
)

# X_transform adalah representasi matriks (numerik) mentah dari artikel teks
X_transform = tfidf.fit_transform(df_target['cleaned_text'])

print(f"\n[*] Dimensi Fitur Transformasi TF-IDF (X) : {X_transform.shape}")
print("[*] 5 Data AFTER Transformation (Matriks TF-IDF) - Menampilkan kolom yang memiliki nilai:")
tfidf_feature_names = tfidf.get_feature_names_out()
df_tfidf = pd.DataFrame(X_transform[:5].toarray(), columns=tfidf_feature_names)
display(df_tfidf.loc[:, df_tfidf.sum() > 0]) # Hanya tampilkan kolom yang bernilai > 0 di 5 data ini

--- MARKDOWN ---
## 6.2 Data Normalization (Skala L2)
Menormalisasi jarak nilai vektor (Scaling L2) pada skor TF-IDF yang sudah diekstrak agar direntangkan secara proporsional dari 0 hingga 1. Ini penting untuk meminimalisasi bias akibat panjang/pendeknya artikel dokumen.

--- CODE ---
from sklearn.preprocessing import normalize

# 2. Lakukan Normalisasi L2 secara manual pada matriks TF-IDF
X = normalize(X_transform, norm='l2')

print("[*] 5 Data AFTER Normalization (Nilai Diskala Menggunakan L2 Norm):")
df_normalized = pd.DataFrame(X[:5].toarray(), columns=tfidf_feature_names)
display(df_normalized)

--- MARKDOWN ---
## 6.3 Data Discretization / Binarisasi Target
Memastikan variabel kelas `y` berupa indikator diskrit/biner (0 atau 1). Karena di dataset aslinya dari Reuters masing-masing label sudah disediakan dalam format biner mutlak (One-Hot Encoding format untuk multi-label), kita hanya perlu memisahkannya secara khusus ke dalam list target `y` tunggal.

--- CODE ---
# 3. Target Discretization (Memastikan y berupa representasi diskrit biner)
y = df_target[agri_labels].values

print(f"[*] Dimensi Target Label Biner (y) : {y.shape}")
print("\n[*] 5 Data Target Label AFTER Discretization (Telah dipisah menjadi matriks target yang utuh):")
df_y = pd.DataFrame(y[:5], columns=agri_labels)
display(df_y)

--- MARKDOWN ---
## 7. Feature Selection (Pembobotan Signifikansi Fitur)
Seringkali data text mining memunculkan "Curse of Dimensionality". Meski sudah direduksi sampai 5000 fitur vektor kalimat, kita perlu memilah dan memilih lagi dimensi utama apa yang paling berguna untuk melatih model kelas-kelas sub-kategori spesifik pertanian. 

Mekanisme **Feature Selection** disini dapat dilakukan menggunakan algoritma `Chi-Square` untuk memeriksa korelasi signifikan antara term (kata) terhadap masing-masing kelas. Coba kita tes memisahkan representasi term untuk kategori berita `'wheat'` (Gandum).

--- CODE ---
from sklearn.feature_selection import SelectKBest, chi2

# Contoh: Seleksi top 15 fitur (Term terbaik) untuk kategori 'wheat' (Gandum)
target_wheat = df_target['wheat'].values

chi2_selector = SelectKBest(score_func=chi2, k=15)
X_kbest = chi2_selector.fit_transform(X_tfidf, target_wheat)

# Mari ambil nama feature words-nya dan urutkan valuenya
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
selected_indices = chi2_selector.get_support(indices=True)

# Ambil score korelasinya
chi2_scores = chi2_selector.scores_[selected_indices]

# Bind dan urutkan dalam DataFrame sederhana
wheat_features_df = pd.DataFrame({
    'Feature (Word)': feature_names[selected_indices],
    'Chi2 Score': chi2_scores
}).sort_values(by='Chi2 Score', ascending=False)

# Visualisasi korelasinya
plt.figure(figsize=(10, 6))
sns.barplot(x='Chi2 Score', y='Feature (Word)', data=wheat_features_df, palette='Oranges_r')
plt.title("Top 15 Feature Selection (Chi-Square) untuk Kategori 'Wheat'")
plt.xlabel('Chi-Square Importance Score')
plt.ylabel('Kata (Fitur)')
plt.show()


--- MARKDOWN ---
### Kesimpulan
Rangkaian Pre-Processing (Cleaning, Integrasi, Normalisasi, Transformasi, dan Feature Selection) telah siap menampung `X_tfidf` dan vektor Label `y_target` menuju tahapan **Modeling Klasifikasi Machine Learning**.

