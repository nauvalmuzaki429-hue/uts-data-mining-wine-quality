# UTS Data Mining - Klasifikasi Kualitas Anggur (Wine Quality)

Proyek ini adalah tugas Ujian Tengah Semester (Take Home Test) mata kuliah **Data Mining**, Program Studi D3 Staterkom, Universitas Negeri Semarang (UNNES), Semester Ganjil 2026/2027.

| | |
|---|---|
| **Nama** | Naufal Muzaky |
| **NIM** | 2404010024 |
| **Dosen Pengampu** | Nur Achmey Selgi Harwanti, M.Stat |

## Deskripsi

Tujuan proyek ini adalah membangun model klasifikasi untuk memprediksi **kualitas anggur** (`quality`, skala 0-10) dari 11 fitur kimiawi, seperti keasaman, kadar alkohol, sulfat, dan densitas. Model dilatih memakai `data_training.csv`, lalu dipakai memprediksi `data_testing.csv`.

## Struktur Repository

```
.
├── UTS_Data_Mining_Naufal_Muzaky.ipynb   # Notebook analisis lengkap (kode + penjelasan + interpretasi)
├── hasilprediksi_024.csv                 # Hasil prediksi data testing (kolom: Id, Quality)
├── model_wine_quality.pkl                # Model Extra Trees yang sudah dilatih
├── data_training.csv                     # Data latih (857 baris)
├── data_testing.csv                      # Data uji tanpa label (286 baris)
├── hasilprediksi_3digitNIMterakhir.csv   # Template format hasil prediksi dari dosen
└── README.md
```

## Alur Analisis

1. **Persiapan data** - memuat data training, data testing, dan template hasil.
2. **EDA** - distribusi target, korelasi antar variabel, distribusi fitur, dan pemeriksaan skala fitur.
3. **Pembersihan data** - pemeriksaan missing values dan duplikat, serta keputusan *feature scaling*.
4. **Pembagian data** - 80% train dan 20% validation (stratified).
5. **Pemodelan** - membandingkan 5 model: KNN, Decision Tree, Random Forest, Extra Trees, dan SVM.
6. **Tuning hyperparameter** - `GridSearchCV` (5-fold) pada Random Forest dan Extra Trees.
7. **Evaluasi** - akurasi, F1-macro, *classification report*, dan *confusion matrix*.
8. **Deployment** - model disimpan ke `.pkl`, dimuat ulang, lalu dipakai memprediksi data testing.
9. **Ekspor hasil** - file `hasilprediksi_024.csv` berisi kolom `Id` dan `Quality`.

## Temuan Utama

- Tidak ada missing values. Ditemukan 76 baris duplikat pada data training.
- Target sangat tidak seimbang: kelas 5 dan 6 mencakup sekitar 82% data, sedangkan kelas 3, 4, dan 8 sangat sedikit.
- Fitur yang paling berkaitan dengan kualitas: **alcohol**, **volatile acidity**, dan **sulphates**.
- Skala antar fitur sangat berbeda, sehingga `StandardScaler` dipakai untuk model berbasis jarak (KNN dan SVM).

### Perbandingan Model (akurasi cross-validation 5-fold)

| Model | Akurasi CV |
|---|---|
| Extra Trees | 0,583 |
| Random Forest | 0,574 |
| SVM | 0,570 |
| Decision Tree | 0,506 |
| KNN | 0,503 |

### Model Terbaik: Extra Trees (setelah tuning)

- Parameter: `n_estimators=500`, `max_depth=15`, `min_samples_leaf=1`, `max_features='sqrt'`
- Akurasi CV: **0,591**
- Akurasi pada data validation: **0,592**

Model bekerja baik pada kelas mayoritas (5 dan 6), tetapi belum mampu memprediksi kelas langka (3, 4, dan 8) karena contoh latihnya sangat sedikit.

## Penanganan Duplikat

- **Evaluasi model:** baris duplikat dibuang (857 menjadi 781 baris) agar skor validasi tidak terlalu optimis.
- **Model final:** dilatih pada seluruh data training, termasuk duplikat, karena 45 dari 286 baris data testing identik dengan baris di data training.

## Cara Menjalankan

### Google Colab
1. Buka notebook `UTS_Data_Mining_Naufal_Muzaky.ipynb` di Google Colab.
2. Jalankan seluruh sel (*Runtime -> Run all*).
3. Jika file CSV belum tersedia, notebook akan meminta upload `data_training.csv`, `data_testing.csv`, dan `hasilprediksi_3digitNIMterakhir.csv`.

### Lokal
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook UTS_Data_Mining_Naufal_Muzaky.ipynb
```
Letakkan file CSV di folder yang sama dengan notebook.

## Menggunakan Model yang Sudah Disimpan

```python
import joblib
import pandas as pd

paket = joblib.load("model_wine_quality.pkl")
model, fitur = paket["model"], paket["fitur"]

data_baru = pd.read_csv("data_testing.csv")
prediksi = model.predict(data_baru[fitur])
```

## Format File Hasil Prediksi

`hasilprediksi_024.csv` hanya berisi dua kolom dengan pemisah titik koma (`;`), sesuai template dari dosen:

```
Id;Quality
222;5
1514;6
...
```

## Teknologi

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, dan joblib.

## Sumber Data

Dataset Wine Quality (anggur merah dan putih) yang disediakan oleh dosen pengampu: <https://bit.ly/datasetwine>
