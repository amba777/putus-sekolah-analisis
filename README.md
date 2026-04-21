# 📊 Analisis Penentu Tingkat Putus Sekolah di Daerah Pedesaan

Dashboard Data Mining & Machine Learning untuk menganalisis faktor-faktor penentu tingkat putus sekolah di daerah pedesaan.

## 👩‍🎓 Kelompok Penelitian

| Nama | NIRM |
|------|------|
| Gladis Primadona | 2024020179 |
| Aulia Virgara | 2024020230 |
| Jesika Tarigan | 2024020119 |

---

## 🎯 Deskripsi Proyek

Proyek ini menganalisis **100 data responden** dari daerah pedesaan untuk mengidentifikasi faktor-faktor yang mempengaruhi tingkat putus sekolah. Faktor yang diteliti meliputi:

| Faktor | Variabel |
|--------|----------|
| 💰 **Ekonomi** | Pendapatan keluarga, kepemilikan lahan, status bansos, jumlah tanggungan |
| 🌿 **Lingkungan** | Kondisi wilayah, akses listrik & internet, fasilitas belajar |
| 👨‍👩‍👧 **Pekerjaan Orang Tua** | Jenis pekerjaan, tingkat pendidikan, jam kerja anak |
| 🛣️ **Jalan & Jarak** | Jarak sekolah, jenis jalan, transportasi, waktu tempuh |
| 💡 **Minat & Motivasi** | Minat belajar, dukungan orang tua, motivasi, pengaruh teman |

## 🤖 Metode Data Mining

Aplikasi ini menggunakan dua algoritma klasifikasi:

### 1. Decision Tree
- Mudah diinterpretasi dan transparan
- Cocok untuk menjelaskan faktor penentu kepada pemangku kebijakan
- Visualisasi pohon keputusan yang intuitif

### 2. Random Forest
- Akurasi lebih tinggi (ensemble method)
- Tahan terhadap overfitting
- Feature importance yang lebih stabil

## 🚀 Cara Menjalankan

### Lokal
```bash
# Clone repository
git clone https://github.com/username/putus-sekolah-analysis.git
cd putus-sekolah-analysis

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

### Deploy ke Streamlit Cloud
1. Push ke GitHub repository
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Hubungkan repository GitHub
4. Set `app.py` sebagai main file
5. Deploy!

## 📁 Struktur Proyek

```
putus_sekolah/
├── app.py                          # Aplikasi Streamlit utama
├── requirements.txt                # Dependencies Python
├── README.md                       # Dokumentasi
├── .streamlit/
│   └── config.toml                 # Konfigurasi tema Streamlit
└── data/
    ├── dataset_putus_sekolah.csv   # Dataset (100 responden)
    └── dataset_putus_sekolah.xlsx  # Dataset format Excel (3 sheet)
```

## 📊 Fitur Aplikasi

- **📊 Eksplorasi Data** — Distribusi, statistik deskriptif, analisis demografi
- **📈 Visualisasi Per Faktor** — Analisis mendalam per kategori faktor
- **🤖 Hasil Model** — Confusion matrix, ROC curve, feature importance, classification report
- **🔮 Prediksi Individual** — Input data baru untuk prediksi risiko putus sekolah
- **📋 Dataset** — Tabel data lengkap dengan filter dan export

## 🛠️ Tech Stack

- **Frontend**: Streamlit + Plotly
- **ML**: Scikit-learn (Decision Tree, Random Forest)
- **Data**: Pandas, NumPy
- **Excel**: OpenPyXL
