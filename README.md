# 🌦️ Komparasi Strategi Imputasi Missing Value pada Dataset Cuaca Australia

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)]()
[![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Repositori ini memuat implementasi kode, eksperimen, dan hasil analisis untuk **Tugas Besar Mata Kuliah Penambangan Data (Data Mining) - Kelompok 1**. Proyek ini berfokus pada analisis *missing value* dan perbandingan performa 5 metode imputasi yang berbeda dalam memprediksi curah hujan di Australia menggunakan algoritma **Random Forest Classifier**.

---

## 📂 Struktur Repositori

Proyek ini terdiri dari direktori dan file utama berikut:

- 📊 **`weatherAUS.csv`** — Dataset mentah yang memuat data observasi cuaca harian dari berbagai lokasi di Australia.
- 📓 **`Tubes_DM_Kelompok_1_Missing_Value (2).ipynb`** — *Jupyter Notebook* yang memuat langkah-langkah detail secara interaktif (Exploratory Data Analysis, visualisasi, identifikasi *missing value*, dan algoritma imputasi).
- 🐍 **`tubes_dm_kelompok_1_missing_value.py`** — *Script* Python fungsional yang berisi alur otomatis untuk pra-pemrosesan data dan penanganan *missing value*.
- 📄 **`PPT_Tubes_DM_Kelompok_1_Missing_Value (1).pdf`** — *Slide* presentasi yang merangkum metodologi, pendekatan, dan hasil temuan dari proyek ini.
- 📑 **`Logbook_Kelompok1_Penambang Data.pdf`** — Catatan *logbook* yang mendokumentasikan proses pengerjaan, pembagian tugas, dan progres mingguan tim.
- 📑 **`Laporan_DM_Kelompok 1_Missing Value.pdf`** — Laporan Hasil riset Kelompok 1 Topik Missing Value dengan dataset Cuaca Australia.
---

## 📊 Tentang Dataset

Dataset yang digunakan adalah **Rain in Australia** (Kaggle/UCI) dengan karakteristik berikut:

- **Dimensi Awal:** 145.460 baris × 23 kolom (22 fitur meteorologi & 1 target)
- **Target Klasifikasi:** `RainTomorrow` (Prediksi apakah besok akan turun hujan: `Yes` atau `No`)
- **Distribusi Target:** *Imbalanced* (77,6% No, 22,4% Yes)
- **Karakteristik Masalah:** Terdapat nilai kosong (*missing value*) pada 21 dari 23 kolom yang ada, dengan persentase kehilangan data tertinggi berada di kolom `Sunshine` (48,01%) dan `Evaporation` (43,17%)

---

## ⚙️ Pipeline Implementasi Kode

Proyek ini dibangun menggunakan Python dengan pustaka utama seperti `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, dan `missingno`. Berikut adalah alur pengerjaan yang diimplementasikan:

### 1. Exploratory Data Analysis (EDA)

- Menganalisis ringkasan nilai kosong dan statistik deskriptif data numerik.
- Memvisualisasikan distribusi target kelas untuk mendeteksi ketidakseimbangan kelas (*class imbalance*).
- Memanfaatkan pustaka `missingno` (bar chart, matriks, dan heatmap korelasi) untuk mengidentifikasi tipe kehilangan data (seperti *Missing at Random* / MAR).

### 2. Preprocessing Data & Pencegahan Data Leakage

- **Penanganan Target NaN:** Menghapus baris yang memiliki nilai kosong pada kolom target `RainTomorrow` sebelum melakukan pembagian data agar tidak merusak proses evaluasi.
- **Encoding Fitur:** Fitur kategorik dikonversi menjadi numerik menggunakan `LabelEncoder`, sementara target `RainTomorrow` dipetakan secara manual (No → 0, Yes → 1).
- **Train-Test Split (80:20):** Dataset dibagi menggunakan metode *Stratified Split* demi menjaga proporsi kelas target tetap seimbang di set data latih dan uji.
- 🔒 **Pencegahan Data Leakage:** Seluruh objek imputer **hanya di-fit pada set data latih (`X_train_raw`)**, kemudian digunakan untuk mentransformasikan data latih maupun data uji (`X_test_raw`).

### 3. Eksperimen 5 Skenario Imputasi

Proyek ini menguji dan membandingkan 5 teknik penanganan *missing value* yang berbeda:

1. **Baseline (Drop NA):** Menghapus seluruh baris yang mengandung nilai kosong (menyisakan 46.466 sampel pada data latih atau membuang ~59,2% data).
2. **Mean Imputation:** Mengisi nilai kosong dengan rata-rata kolom dari data latih.
3. **Median Imputation:** Mengisi nilai kosong dengan nilai tengah kolom (lebih kebal terhadap *outlier*).
4. **KNN Imputer (k=5):** Mengimputasi nilai kosong menggunakan rata-rata tertimbang dari 5 tetangga terdekat berdasarkan jarak Euclidean.
5. **MICE (Iterative Imputer):** Menggunakan teknik imputasi multivariat berantai, di mana setiap fitur dengan *missing value* dimodelkan secara iteratif (`max_iter=10`) menggunakan fitur lain sebagai prediktor.

---

## 📈 Hasil Evaluasi Model

Setiap skenario imputasi diuji menggunakan model **Random Forest Classifier** (`n_estimators=100`, `random_state=42`, `n_jobs=-1`). Berikut adalah rangkuman performa dari kelima skenario tersebut:

| Metode Imputasi | N Train | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Waktu Training |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Drop NA)** | 46.466 | **0.8625** | **0.8554** | **0.8625** | **0.8530** | **0.8979** | **17.16s** |
| **Mean** | 113.754 | 0.8569 | 0.8493 | 0.8569 | 0.8459 | 0.8884 | 30.80s |
| **Median** | 113.754 | 0.8571 | 0.8496 | 0.8571 | 0.8463 | 0.8882 | 30.90s |
| **KNN (k=5)** | 113.754 | 0.8560 | 0.8483 | 0.8560 | 0.8448 | 0.8847 | 34.72s* |
| **MICE** | 113.754 | 0.8567 | 0.8491 | 0.8567 | 0.8459 | 0.8869 | 48.65s |

> *\*Catatan: Meskipun waktu training model KNN relatif cepat (34.72s), proses imputasi data menggunakan KNN Imputer membutuhkan waktu yang paling lama, yaitu mencapai **1371.38 detik** (~22,8 menit).*

### 🔑 Kesimpulan Utama Eksperimen

Metode **Baseline (Drop NA)** menghasilkan performa tertinggi pada semua metrik evaluasi (F1-Score ~0.8530 dan AUC-ROC ~0.8979). Hal ini menunjukkan bahwa pada dataset ini, memaksakan pengisian data (*imputation*) pada kolom yang kehilangan datanya hampir mencapai 50% (seperti `Sunshine` dan `Evaporation`) justru memperkenalkan *noise* baru yang mengaburkan pola klasifikasi model. Menghapus baris yang tidak lengkap terbukti menghasilkan subset data yang lebih bersih dan optimal bagi algoritma Random Forest.

---

## 🚀 Cara Menjalankan Proyek

**1. Clone repositori ini:**

```bash
git clone https://github.com/garsrayy/datamining2026.git
cd datamining2026
```

**2. Instalasi Pustaka Pendukung:**

Pastikan Python telah terinstal di komputer Anda. Jalankan perintah berikut untuk menginstal seluruh dependensi:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn missingno jupyter
```

**3. Menjalankan Analisis:**

- Untuk menjalankan notebook secara interaktif, ketik perintah berikut lalu buka file `Tubes_DM_Kelompok_1_Missing_Value (2).ipynb`:
  ```bash
  jupyter notebook
  ```
- Untuk mengeksekusi pipeline otomatis lewat terminal, jalankan:
  ```bash
  python tubes_dm_kelompok_1_missing_value.py
  ```

---

## 👥 Tim Kolaborator (Kelompok 1)

Proyek riset dan implementasi ini diselesaikan secara berkelompok oleh:

| Nama | Username GitHub | Tugas / Kontribusi Utama |
| :--- | :--- | :--- |
| **Garis Rayya Rabbani** | [@garsrayy](https://github.com/garsrayy) | Ketua Kelompok / Latar Belakang & Implementasi Kode |
| **Refi Ikhsanti** | [@7refisa](https://github.com/7refisa) | Tinjauan Pustaka – Mekanisme Missing Value |
| **Choirunnisa Syawaldina** | [@choirunnisasy](https://github.com/choirunnisasy) | Tinjauan Pustaka – Strategi Imputasi & Metrik Evaluasi |
| **Anisah Octa Rohila** | [@panesbelpois](https://github.com/panesbelpois) | Rumusan Masalah & Preprocessing Data |
| **Awi Septian Prasetyo** | [@Awesome1209](https://github.com/Awesome1209) | Implementasi Kode & Pemodelan Random Forest |
| **Muhammad Bimastiar** | [@211-Bimas](https://github.com/211-Bimas) | Analisis Hasil Eksperimen & Penulisan Kesimpulan/Saran |

---

*Dibuat untuk memenuhi Tugas Besar Mata Kuliah Penambangan Data IF25-32025 Tahun Akademik 2025/2026.*
