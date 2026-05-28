# 🩺 Diabetes Prediction App

Aplikasi berbasis web sederhana namun interaktif untuk memprediksi probabilitas seseorang menderita **diabetes** berdasarkan parameter klinis dan gaya hidup mereka. Projek ini menggunakan algoritma **Machine Learning (Random Forest Classifier)** yang terintegrasi dengan antarmuka web interaktif berbasis **Streamlit**.

---

## 🚀 Fitur Utama

- **Prediksi Interaktif:** Antarmuka web yang intuitif menggunakan Streamlit untuk memasukkan data pasien secara mudah.
- **Probabilitas Hasil:** Menampilkan persentase kemungkinan diagnosis diabetes secara langsung beserta status kelayakan klinis.
- **Visualisasi Hasil:** Pie chart dinamis yang memvisualisasikan tingkat probabilitas diabetes vs non-diabetes.
- **Pipeline Preprocessing:** Integrasi standardisasi skala fitur numerik (`StandardScaler`) dan enkoding kategori (`OneHotEncoder`) menggunakan `ColumnTransformer` scikit-learn.
- **Exploratory Data Analysis (EDA):** Visualisasi sebaran label, confusion matrix, dan heatmap korelasi fitur tersimpan otomatis selama proses training.

---

## 📊 Detail Dataset & Fitur

Dataset yang digunakan untuk melatih model ini tersimpan di `data/diabetes_data.csv` dengan fitur-fitur sebagai berikut:

### Fitur Numerik (Skala StandardScaler)
1. **Age (Usia):** Usia pasien dalam satuan tahun (0–120).
2. **BMI (Body Mass Index):** Indeks massa tubuh pasien (10.0–60.0).
3. **HbA1c Level:** Kadar hemoglobin A1c pasien, indikator rata-rata gula darah 3 bulan terakhir (3.0–15.0).
4. **Blood Glucose Level:** Kadar glukosa darah pasien saat diperiksa (50–500 mg/dL).

### Fitur Kategorikal (Enkoding OneHotEncoder)
5. **Gender:** Jenis kelamin pasien (`Male`, `Female`, `Other`).
6. **Smoking History:** Riwayat merokok (`never`, `current`, `former`, `not current`, `ever`, `No Info`).

### Fitur Biner (Pass-Through)
7. **Hypertension:** Apakah pasien memiliki hipertensi (`0` = Tidak, `1` = Ya).
8. **Heart Disease:** Apakah pasien memiliki riwayat penyakit jantung (`0` = Tidak, `1` = Ya).

### Target Label
- **Diabetes:** Status diagnosis (`0` = Negatif/Tidak Diabetes, `1` = Positif/Diabetes).

---

## 📁 Struktur Folder Projek

```text
diabetes_prediction/
├── app/
│   └── streamlit_app.py             # Kode aplikasi antarmuka Streamlit
├── data/
│   └── diabetes_data.csv            # Dataset pelatihan model
├── models/
│   └── diabetes_model.pkl           # Model terlatih (pipeline.pkl) yang diekspor
├── notebooks/
│   └── eda_and_training.py          # Script analisis data eksploratif (EDA) & training
├── output_confusion_matrix.png     # Gambar confusion matrix hasil evaluasi model
├── output_correlation.png          # Gambar heatmap korelasi antar fitur numerik
├── output_label_distribution.png   # Gambar distribusi kelas target (Diabetes vs Sehat)
├── train_model.py                   # Script ringkas untuk memproses ulang & melatih model
├── requirements.txt                 # Dependensi pustaka Python
└── README.md                        # Dokumentasi projek ini
```

---

## 🛠️ Instalasi & Cara Menjalankan

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi secara lokal:

### 1. Kloning Projek / Buka Direktori
Masuk ke dalam folder projek utama Anda:
```bash
cd diabetes_prediction
```

### 2. Buat & Aktifkan Virtual Environment (Direkomendasikan)
```bash
# Membuat virtual environment
python -m venv myenv

# Mengaktifkan di Windows (PowerShell)
.\myenv\Scripts\Activate.ps1

# Atau mengaktifkan di Windows (Command Prompt)
.\myenv\Scripts\activate.bat

# Mengaktifkan di macOS/Linux
source myenv/bin/activate
```

### 3. Instal Dependensi
Pasang semua library Python yang dibutuhkan yang tertera pada berkas `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi Streamlit
Jalankan server Streamlit lokal untuk membuka antarmuka web:
```bash
streamlit run app/streamlit_app.py
```
Aplikasi secara otomatis akan terbuka di browser Anda pada alamat `http://localhost:8501`.

---

## 🧠 Proses Training & Evaluasi Model

Jika Anda ingin melatih ulang model dengan parameter atau dataset baru, Anda dapat menjalankan script pelatihan:

```bash
python train_model.py
```

Untuk melihat laporan evaluasi performa model secara lengkap beserta grafik analisis data eksploratif (EDA), jalankan berkas script yang ada di folder `notebooks`:

```bash
python notebooks/eda_and_training.py
```

### Output Evaluasi & Visualisasi
Setelah script `eda_and_training.py` dijalankan, tiga visualisasi penting akan diperbarui secara otomatis di direktori utama:
1. **`output_label_distribution.png`**: Distribusi frekuensi data pasien yang sehat vs penderita diabetes.
2. **`output_correlation.png`**: Korelasi Pearson antar-fitur numerik untuk memetakan ketergantungan variabel.
3. **`output_confusion_matrix.png`**: Matriks konfusi hasil prediksi model pada data pengujian (test set) untuk mengukur tingkat akurasi, sensitivitas, dan spesifisitas model.

---

## 🧪 Teknologi yang Digunakan

Projek ini dibangun menggunakan komponen-komponen unggulan berikut:

- **Bahasa Pemrograman:** [Python 3.x](https://www.python.org/)
- **Antarmuka Pengguna:** [Streamlit](https://streamlit.io/) (Untuk dashboard web interaktif yang modern dan responsif)
- **Komputasi & Manipulasi Data:** [Pandas](https://pandas.pydata.org/)
- **Machine Learning & Pipeline:** [Scikit-Learn](https://scikit-learn.org/) (Algoritma Random Forest Classifier, StandardScaler, OneHotEncoder, ColumnTransformer)
- **Penyimpanan Model:** [Joblib](https://joblib.readthedocs.io/)
- **Visualisasi Grafik:** [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)
