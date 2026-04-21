import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_curve, auc)
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Analisis Putus Sekolah Pedesaan 2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1422 50%, #13182a 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #0a0e1a 100%) !important; border-right: 1px solid rgba(255,215,0,0.2); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,[data-testid="stSidebar"] .stRadio label,[data-testid="stSidebar"] .stSlider label { color: #ffd700 !important; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
.main-header { background: linear-gradient(135deg, #1a1f2e 0%, #0f1422 50%, #1a1f2e 100%); padding: 2rem 2.5rem; border-radius: 20px; margin-bottom: 2rem; border: 1px solid rgba(255,215,0,0.15); }
.main-title { font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, #ffffff 0%, #ffd700 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.5rem; }
.main-subtitle { font-size: 0.9rem; color: #94a3b8 !important; }
.metric-card { background: rgba(18, 22, 35, 0.9); backdrop-filter: blur(10px); border-radius: 16px; padding: 1.2rem; text-align: center; border: 1px solid rgba(255,215,0,0.15); transition: all 0.3s ease; }
.metric-card:hover { transform: translateY(-3px); border-color: rgba(255,215,0,0.4); }
.metric-value { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #ffffff, #ffd700); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.metric-label { font-size: 0.7rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; }
.metric-badge { display: inline-block; font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 20px; margin-top: 0.5rem; }
.badge-danger  { background: rgba(220,38,38,0.2);  color: #f87171; }
.badge-success { background: rgba(34,197,94,0.2);  color: #4ade80; }
.badge-info    { background: rgba(59,130,246,0.2); color: #60a5fa; }
.badge-warning { background: rgba(245,158,11,0.2); color: #fbbf24; }
.section-header { font-size: 1.3rem; font-weight: 700; background: linear-gradient(135deg, #ffffff, #ffd700); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; border-left: 4px solid #ffd700; padding-left: 1rem; margin: 1.5rem 0 1rem 0; }
.section-sub { font-size: 0.85rem; color: #94a3b8; margin-bottom: 1.2rem; }
.stTabs [data-baseweb="tab-list"] { background: #0d1117; border-radius: 12px 12px 0 0; padding: 0.3rem 0.3rem 0; gap: 4px; border-bottom: 1px solid rgba(255,215,0,0.2); }
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; font-weight: 500; font-size: 0.85rem; padding: 0.6rem 1.2rem; border-radius: 8px 8px 0 0; background: transparent !important; }
.stTabs [aria-selected="true"] { color: #ffd700 !important; background: rgba(255,215,0,0.1) !important; border-bottom: 2px solid #ffd700 !important; }
.stTabs [data-baseweb="tab-panel"] { background: rgba(18, 22, 35, 0.6); backdrop-filter: blur(5px); border-radius: 0 12px 12px 12px; padding: 1.5rem; border: 1px solid rgba(255,215,0,0.1); }
.info-box { background: rgba(59,130,246,0.1); border-left: 4px solid #3b82f6; padding: 1rem 1.2rem; border-radius: 8px; margin: 1rem 0; font-size: 0.85rem; color: #cbd5e1; }
.pred-result { padding: 1.5rem; border-radius: 16px; text-align: center; margin: 1rem 0; }
.pred-danger { background: linear-gradient(135deg, rgba(220,38,38,0.2), rgba(220,38,38,0.1)); border: 2px solid #f87171; }
.pred-safe { background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05)); border: 1px solid rgba(34,197,94,0.3); }
.pred-title { font-size: 1.5rem; font-weight: 700; }
.pred-danger .pred-title { color: #f87171; }
.pred-safe .pred-title   { color: #4ade80; }
.stButton > button { background: linear-gradient(135deg, #ffd700, #b8860b); color: #0a0e1a !important; border: none; border-radius: 10px; padding: 0.6rem 1.5rem; font-weight: 700; transition: all 0.2s; }
.stButton > button:hover { background: linear-gradient(135deg, #ffed4a, #daa520); transform: translateY(-2px); }
.footer { background: linear-gradient(135deg, #0d1117, #0a0e1a); color: #94a3b8; padding: 1.5rem; border-radius: 16px; margin-top: 2rem; text-align: center; font-size: 0.75rem; border-top: 1px solid rgba(255,215,0,0.2); }
.footer .group-name { color: #ffd700; font-weight: 600; font-size: 0.85rem; }
.gold-divider { height: 1px; background: linear-gradient(90deg, transparent, #ffd700, transparent); margin: 1.5rem 0; }
.warning-box { background: rgba(245,158,11,0.12); border-left: 4px solid #fbbf24; padding: 0.8rem 1rem; border-radius: 8px; margin: 0.8rem 0; font-size: 0.83rem; color: #fde68a; }
</style>
""", unsafe_allow_html=True)

# ─── MAPPING ───────────────────────────────────────────────────────────────────
MAPPINGS = {
    'Jenis_Kelamin':                  {"Perempuan": 0, "Laki-laki": 1},
    'Kondisi_Lingkungan':             {"Sangat Terpencil": 0, "Terpencil": 1, "Cukup Terjangkau": 2, "Terjangkau": 3},
    'Akses_Listrik':                  {"Tidak": 0, "Ya": 1},
    'Akses_Internet':                 {"Tidak": 0, "Ya": 1},
    'Ketersediaan_Fasilitas_Belajar': {"Sangat Kurang": 0, "Kurang": 1, "Cukup": 2, "Baik": 3},
    'Kepemilikan_Lahan':              {"Tidak Punya": 0, "Sewa": 1, "Milik Sendiri": 2},
    'Status_Bansos':                  {"Tidak": 0, "Ya": 1},
    'Pekerjaan_Ayah': {
        "Ibu Rumah Tangga": 0, "Tidak Bekerja": 1, "Buruh": 2, "Buruh Harian": 3,
        "Nelayan": 4, "Pedagang Kecil": 5, "Pekerja Serabutan": 6, "Petani": 7
    },
    'Pekerjaan_Ibu': {
        "Tidak Bekerja": 0, "Buruh": 1, "Ibu Rumah Tangga": 2, "Pedagang Kecil": 3, "Petani": 4
    },
    'Pendidikan':    {"Tidak Sekolah": 0, "SD": 1, "SMP": 2, "SMA": 3, "Diploma/S1": 4},
    'Jenis_Jalan':   {"Jalan Tanah": 0, "Jalan Kerikil": 1, "Jalan Aspal Rusak": 2, "Jalan Aspal Baik": 3},
    'Ketersediaan_Transportasi': {"Tidak Ada": 0, "Ojek/Becak": 1, "Angkutan Umum": 2, "Kendaraan Pribadi": 3},
    'Kondisi_Jalan_Saat_Hujan':  {"Tidak Bisa Dilalui": 0, "Sangat Sulit": 1, "Sulit": 2, "Bisa Dilalui": 3},
    'Skala5': {"Sangat Rendah": 0, "Rendah": 1, "Sedang": 2, "Tinggi": 3, "Sangat Tinggi": 4},
    'Dukungan_Orang_Tua': {"Sangat Kurang": 0, "Kurang": 1, "Cukup": 2, "Baik": 3, "Sangat Baik": 4},
    'Pengaruh_Teman_Sebaya': {"Sangat Negatif": 0, "Negatif": 1, "Netral": 2, "Positif": 3, "Sangat Positif": 4},
    'Kelas_Terakhir': {k: i for i, k in enumerate([
        "SD Kelas 1","SD Kelas 2","SD Kelas 3","SD Kelas 4","SD Kelas 5","SD Kelas 6",
        "SMP Kelas 7","SMP Kelas 8","SMP Kelas 9","SMA Kelas 10","SMA Kelas 11","SMA Kelas 12"
    ])},
}

FEATURE_COLS = [
    'Usia_Anak', 'Jenis_Kelamin_Enc', 'Kelas_Terakhir_Enc',
    'Pendapatan_Keluarga_Bulan', 'Jumlah_Tanggungan_Keluarga',
    'Kepemilikan_Lahan_Enc', 'Status_Bansos_Enc',
    'Kondisi_Lingkungan_Enc', 'Akses_Listrik_Enc', 'Akses_Internet_Enc',
    'Ketersediaan_Fasilitas_Belajar_Enc', 'Pekerjaan_Ayah_Enc', 'Pekerjaan_Ibu_Enc',
    'Pendidikan_Ayah_Enc', 'Pendidikan_Ibu_Enc', 'Jam_Kerja_Anak_Per_Minggu',
    'Jarak_ke_Sekolah_km', 'Jenis_Jalan_Enc', 'Waktu_Tempuh_Menit',
    'Ketersediaan_Transportasi_Enc', 'Kondisi_Jalan_Saat_Hujan_Enc',
    'Minat_Belajar_Anak_Enc', 'Dukungan_Orang_Tua_Enc',
    'Motivasi_Melanjutkan_Sekolah_Enc', 'Pengaruh_Teman_Sebaya_Enc'
]

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_putus_sekolah.csv")

@st.cache_data
def prepare_model_data(df):
    d = df.copy()
    d['Jenis_Kelamin_Enc']                  = d['Jenis_Kelamin'].map(MAPPINGS['Jenis_Kelamin'])
    d['Kelas_Terakhir_Enc']                 = d['Kelas_Terakhir'].map(MAPPINGS['Kelas_Terakhir'])
    d['Kepemilikan_Lahan_Enc']              = d['Kepemilikan_Lahan'].map(MAPPINGS['Kepemilikan_Lahan'])
    d['Status_Bansos_Enc']                  = d['Status_Bansos'].map(MAPPINGS['Status_Bansos'])
    d['Kondisi_Lingkungan_Enc']             = d['Kondisi_Lingkungan'].map(MAPPINGS['Kondisi_Lingkungan'])
    d['Akses_Listrik_Enc']                  = d['Akses_Listrik'].map(MAPPINGS['Akses_Listrik'])
    d['Akses_Internet_Enc']                 = d['Akses_Internet'].map(MAPPINGS['Akses_Internet'])
    d['Ketersediaan_Fasilitas_Belajar_Enc'] = d['Ketersediaan_Fasilitas_Belajar'].map(MAPPINGS['Ketersediaan_Fasilitas_Belajar'])
    d['Pekerjaan_Ayah_Enc']                 = d['Pekerjaan_Ayah'].map(MAPPINGS['Pekerjaan_Ayah'])
    d['Pekerjaan_Ibu_Enc']                  = d['Pekerjaan_Ibu'].map(MAPPINGS['Pekerjaan_Ibu'])
    d['Pendidikan_Ayah_Enc']                = d['Pendidikan_Ayah'].map(MAPPINGS['Pendidikan'])
    d['Pendidikan_Ibu_Enc']                 = d['Pendidikan_Ibu'].map(MAPPINGS['Pendidikan'])
    d['Jenis_Jalan_Enc']                    = d['Jenis_Jalan'].map(MAPPINGS['Jenis_Jalan'])
    d['Ketersediaan_Transportasi_Enc']      = d['Ketersediaan_Transportasi'].map(MAPPINGS['Ketersediaan_Transportasi'])
    d['Kondisi_Jalan_Saat_Hujan_Enc']       = d['Kondisi_Jalan_Saat_Hujan'].map(MAPPINGS['Kondisi_Jalan_Saat_Hujan'])
    d['Minat_Belajar_Anak_Enc']             = d['Minat_Belajar_Anak'].map(MAPPINGS['Skala5'])
    d['Dukungan_Orang_Tua_Enc']             = d['Dukungan_Orang_Tua'].map(MAPPINGS['Dukungan_Orang_Tua'])
    d['Motivasi_Melanjutkan_Sekolah_Enc']   = d['Motivasi_Melanjutkan_Sekolah'].map(MAPPINGS['Skala5'])
    d['Pengaruh_Teman_Sebaya_Enc']          = d['Pengaruh_Teman_Sebaya'].map(MAPPINGS['Pengaruh_Teman_Sebaya'])
    X = d[FEATURE_COLS]
    y = d['Label']
    return X, y

df = load_data()
X, y = prepare_model_data(df)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.5rem 0'>
        <div style='font-size:2.5rem'>🎓</div>
        <div style='font-size:1rem; font-weight:700; color:#ffd700; margin-top:0.3rem'>Analisis Putus Sekolah</div>
        <div style='font-size:0.7rem; color:#94a3b8'>Data Mining Dashboard • 2026</div>
        <div style='height:1px; background:linear-gradient(90deg,transparent,#ffd700,transparent); margin:1rem 0'></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem; color:#ffd700; font-weight:600; letter-spacing:0.1em'>⚙️ MODEL CONFIGURATION</p>", unsafe_allow_html=True)
    st.info("📌 **Metode:** Random Forest\n\nAkurasi tinggi & tahan overfitting")
    test_size = st.slider("Test Split (%)", 20, 40, 30, 5)
    max_depth = st.slider("Max Depth",       2, 10,  5)
    n_trees   = st.slider("Jumlah Pohon (n_estimators)", 50, 200, 100, 25)
    st.markdown("<div style='height:1px; background:linear-gradient(90deg,transparent,#ffd700,transparent); margin:1rem 0'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem; color:#ffd700; font-weight:600; letter-spacing:0.1em'>📋 KELOMPOK PENELITI</p>", unsafe_allow_html=True)
    for icon, name, nirm in [
        ("👩‍🎓", "Gladis Primadona", "2024020179"),
        ("👩‍🎓", "Aulia Virgara",    "2024020230"),
        ("👩‍🎓", "Jesika Tarigan",   "2024020119"),
    ]:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03); border-radius:10px; padding:0.5rem 0.7rem;
                    margin-bottom:0.5rem; border-left:2px solid #ffd700'>
            <div style='font-size:0.8rem; font-weight:600; color:#ffd700'>{icon} {name}</div>
            <div style='font-size:0.65rem; color:#94a3b8'>NIRM: {nirm}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Train Model ───────────────────────────────────────────────────────────────
@st.cache_data
def train_model(test_sz, depth, n_est):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_sz/100, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth, random_state=42, class_weight='balanced')
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    cv  = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    fi  = pd.DataFrame({'feature': FEATURE_COLS, 'importance': model.feature_importances_})
    fi  = fi.sort_values('importance', ascending=False)
    return model, X_te, y_te, y_pred, y_prob, acc, cv, fi

model, X_te, y_te, y_pred, y_prob, acc, cv, fi = train_model(test_size, max_depth, n_trees)

# ─── Fungsi Risk Score Berbasis Aturan ─────────────────────────────────────────
def hitung_rule_risk_score(inp):
    """
    Menghitung skor risiko berbasis aturan domain (0–100).
    Dirancang untuk menangkap kasus ekstrem yang memiliki banyak faktor risiko
    namun jam kerja rendah (di luar distribusi training data).
    """
    skor = 0.0

    # 1. Ekonomi (maks 25)
    if inp['pendapatan'] < 500_000:   skor += 25
    elif inp['pendapatan'] < 800_000:  skor += 18
    elif inp['pendapatan'] < 1_200_000: skor += 10
    elif inp['pendapatan'] < 1_800_000: skor += 5

    # 2. Beban kerja anak (maks 20) — lebih proporsional, tidak cliff di 15
    jk = inp['jam_kerja']
    if jk >= 25:   skor += 20
    elif jk >= 18: skor += 16
    elif jk >= 12: skor += 10
    elif jk >= 6:  skor += 5
    elif jk >= 3:  skor += 2

    # 3. Psikologi & dukungan (maks 20)
    # minat [0-4], motivasi [0-4], dukungan [0-4], teman [0-4] → total maks 16
    psi = inp['minat'] + inp['motivasi'] + inp['dukungan'] + inp['teman']
    skor += (1 - psi / 16) * 20

    # 4. Akses transportasi & jarak (maks 15)
    if inp['jarak'] > 25:    skor += 8
    elif inp['jarak'] > 15:  skor += 5
    elif inp['jarak'] > 8:   skor += 2
    if inp['transportasi'] == 0:   skor += 7
    elif inp['transportasi'] == 1: skor += 3

    # 5. Lingkungan (maks 12)
    if inp['kondisi_ling'] == 0:   skor += 5
    elif inp['kondisi_ling'] == 1: skor += 2
    if inp['listrik'] == 0:        skor += 3
    if inp['internet'] == 0:       skor += 2
    if inp['fasilitas'] == 0:      skor += 2

    # 6. Tanggungan keluarga (maks 8)
    t = inp['tanggungan']
    if t >= 8:   skor += 8
    elif t >= 6: skor += 5
    elif t >= 4: skor += 2

    # Total maks teoritis ≈ 100
    return min(skor, 100.0)

# ─── Header ────────────────────────────────────────────────────────────────────
total     = len(df)
putus     = (df['Status_Putus_Sekolah'] == 'Ya').sum()
tidak     = (df['Status_Putus_Sekolah'] == 'Tidak').sum()
pct_putus = putus / total * 100

st.markdown("""
<div class="main-header">
    <div class="main-title">📊 Analisis Penentu Tingkat Putus Sekolah</div>
    <div class="main-subtitle">Di Daerah Pedesaan · Random Forest Classification · Studi Kasus 100 Responden · 2026</div>
</div>
""", unsafe_allow_html=True)

RED   = "#f87171"
GREEN = "#4ade80"

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, lbl, badge, sub in [
    (c1, str(total),         "Total Responden",  "badge-info",    "📁 Dataset 2026"),
    (c2, str(putus),         "Putus Sekolah",    "badge-danger",  f"⚠️ {pct_putus:.1f}% dari total"),
    (c3, str(tidak),         "Tidak Putus",      "badge-success", f"✅ {100-pct_putus:.1f}% dari total"),
    (c4, f"{acc*100:.1f}%",  "Akurasi Model",    "badge-info",    "🤖 Random Forest"),
    (c5, f"{cv*100:.1f}%",   "CV Akurasi",       "badge-warning", "🔁 5-Fold Cross Val"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
            <span class="metric-badge {badge}">{sub}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📖 Tentang Metode",
    "📊 Eksplorasi Data",
    "📈 Visualisasi Faktor",
    "🤖 Hasil Model",
    "🔮 Prediksi",
    "📋 Dataset"
])

# ══════ TAB 0 ══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div class='section-header'>🌲 Tentang Random Forest</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class='info-box' style='background:rgba(255,215,0,0.05); border-left-color:#ffd700;'>
        <b>📌 Apa itu Random Forest?</b><br>
        Random Forest adalah algoritma <b>ensemble learning</b> yang menggabungkan banyak
        <b>Decision Tree</b> untuk menghasilkan prediksi yang lebih akurat dan stabil.
        </div>
        <div class='info-box' style='background:rgba(59,130,246,0.05); border-left-color:#3b82f6; margin-top:1rem'>
        <b>⚙️ Cara Kerja:</b><br>
        1. Membuat <b>N pohon keputusan</b> dari sampel data acak (bootstrap)<br>
        2. Setiap pohon memilih <b>fitur secara acak</b> untuk split terbaik<br>
        3. Setiap pohon memberikan <b>prediksi independen</b><br>
        4. Hasil akhir = <b>voting mayoritas</b> dari semua pohon<br>
        5. Feature importance dihitung dari rata-rata penurunan impurity (Gini/Entropy)
        </div>
        <div class='info-box' style='background:rgba(255,215,0,0.05); border-left-color:#ffd700; margin-top:1rem'>
        <b>🔀 Sistem Prediksi Hybrid (Tab Prediksi)</b><br>
        Karena dataset terbatas (100 data), prediksi menggunakan pendekatan <b>hybrid</b>:<br>
        • <b>40%</b> dari probabilitas model Random Forest<br>
        • <b>60%</b> dari skor risiko berbasis aturan domain (domain knowledge)<br>
        Ini memastikan kasus dengan banyak faktor risiko tetap terdeteksi dengan benar,
        bahkan jika salah satu fitur berada di luar distribusi data latih.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:rgba(255,215,0,0.1); border-radius:16px; padding:1rem;
                    text-align:center; border:1px solid rgba(255,215,0,0.3)'>
            <div style='font-size:3rem'>🌲🌲🌲</div>
            <div style='font-size:1.2rem; font-weight:700; color:#ffd700; margin:0.5rem 0'>Random Forest</div>
            <hr style='border-color:rgba(255,215,0,0.2)'>
            <div style='text-align:left; font-size:0.75rem; color:#cbd5e1'>
            ✅ Akurasi tinggi<br>✅ Tahan overfitting<br>
            ✅ Feature importance<br>✅ Handle data kategorikal<br>
            ✅ Robust terhadap outlier
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════ TAB 1 ══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-header'>📊 Distribusi & Gambaran Umum Data</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = go.Figure(go.Pie(
            labels=['Putus Sekolah', 'Tidak Putus'],
            values=[putus, tidak], hole=0.55,
            marker_colors=[RED, GREEN],
            textfont=dict(size=13, color='white'),
        ))
        fig_pie.add_annotation(text=f"<b>{pct_putus:.0f}%</b><br>Putus",
                               x=0.5, y=0.5, font=dict(size=14, color='white'), showarrow=False)
        fig_pie.update_layout(title="Distribusi Status Putus Sekolah",
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=340)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_age = px.histogram(df, x='Usia_Anak', color='Status_Putus_Sekolah',
                               color_discrete_map={'Ya': RED, 'Tidak': GREEN},
                               nbins=11, barmode='overlay', opacity=0.8)
        fig_age.update_layout(title="Distribusi Usia Responden",
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=340)
        st.plotly_chart(fig_age, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_kelas = px.histogram(df, x='Kelas_Terakhir', color='Status_Putus_Sekolah',
                                 color_discrete_map={'Ya': RED, 'Tidak': GREEN},
                                 barmode='group',
                                 category_orders={'Kelas_Terakhir': list(MAPPINGS['Kelas_Terakhir'].keys())})
        fig_kelas.update_layout(title="Distribusi Kelas Terakhir",
                                paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=340,
                                xaxis_tickangle=-45)
        st.plotly_chart(fig_kelas, use_container_width=True)
    with col4:
        avg_income = df.groupby('Status_Putus_Sekolah')['Pendapatan_Keluarga_Bulan'].mean().reset_index()
        fig_inc = px.bar(avg_income, x='Status_Putus_Sekolah', y='Pendapatan_Keluarga_Bulan',
                         color='Status_Putus_Sekolah',
                         color_discrete_map={'Ya': RED, 'Tidak': GREEN})
        fig_inc.update_layout(title="Rata-rata Pendapatan Keluarga",
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=340)
        st.plotly_chart(fig_inc, use_container_width=True)

# ══════ TAB 2 ══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-header'>📈 Analisis Per Faktor Penentu</div>", unsafe_allow_html=True)
    sub_tabs = st.tabs(["💰 Ekonomi", "🌿 Lingkungan", "🛣️ Jalan & Jarak", "💡 Minat & Motivasi"])

    with sub_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x='Status_Putus_Sekolah', y='Pendapatan_Keluarga_Bulan',
                         color='Status_Putus_Sekolah', color_discrete_map={'Ya': RED, 'Tidak': GREEN})
            fig.update_layout(title="Distribusi Pendapatan Keluarga",
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(df, x='Jumlah_Tanggungan_Keluarga', color='Status_Putus_Sekolah',
                                color_discrete_map={'Ya': RED, 'Tidak': GREEN}, barmode='group')
            fig2.update_layout(title="Jumlah Tanggungan Keluarga",
                               paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig2, use_container_width=True)

    with sub_tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            grp = df.groupby(['Kondisi_Lingkungan', 'Status_Putus_Sekolah']).size().reset_index(name='Jumlah')
            fig = px.bar(grp, x='Kondisi_Lingkungan', y='Jumlah', color='Status_Putus_Sekolah',
                         barmode='group', color_discrete_map={'Ya': RED, 'Tidak': GREEN})
            fig.update_layout(title="Kondisi Lingkungan vs Status Putus",
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            grp2 = df.groupby(['Akses_Internet', 'Status_Putus_Sekolah']).size().reset_index(name='Jumlah')
            fig2 = px.bar(grp2, x='Akses_Internet', y='Jumlah', color='Status_Putus_Sekolah',
                          barmode='group', color_discrete_map={'Ya': RED, 'Tidak': GREEN})
            fig2.update_layout(title="Akses Internet vs Status Putus",
                               paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig2, use_container_width=True)

    with sub_tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x='Jarak_ke_Sekolah_km', y='Waktu_Tempuh_Menit',
                             color='Status_Putus_Sekolah',
                             color_discrete_map={'Ya': RED, 'Tidak': GREEN}, opacity=0.7)
            fig.update_layout(title="Jarak vs Waktu Tempuh",
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            grp = df.groupby(['Jenis_Jalan', 'Status_Putus_Sekolah']).size().reset_index(name='Jumlah')
            fig2 = px.bar(grp, x='Jenis_Jalan', y='Jumlah', color='Status_Putus_Sekolah',
                          barmode='group', color_discrete_map={'Ya': RED, 'Tidak': GREEN})
            fig2.update_layout(title="Jenis Jalan vs Status Putus",
                               paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig2, use_container_width=True)

    with sub_tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            grp = df.groupby(['Minat_Belajar_Anak', 'Status_Putus_Sekolah']).size().reset_index(name='Jumlah')
            fig = px.bar(grp, x='Minat_Belajar_Anak', y='Jumlah', color='Status_Putus_Sekolah',
                         barmode='group', color_discrete_map={'Ya': RED, 'Tidak': GREEN},
                         category_orders={'Minat_Belajar_Anak': list(MAPPINGS['Skala5'].keys())})
            fig.update_layout(title="Minat Belajar vs Status Putus",
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            grp2 = df.groupby(['Dukungan_Orang_Tua', 'Status_Putus_Sekolah']).size().reset_index(name='Jumlah')
            fig2 = px.bar(grp2, x='Dukungan_Orang_Tua', y='Jumlah', color='Status_Putus_Sekolah',
                          barmode='group', color_discrete_map={'Ya': RED, 'Tidak': GREEN},
                          category_orders={'Dukungan_Orang_Tua': list(MAPPINGS['Dukungan_Orang_Tua'].keys())})
            fig2.update_layout(title="Dukungan Orang Tua vs Status Putus",
                               paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig2, use_container_width=True)

# ══════ TAB 3 ══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-header'>🤖 Hasil Model: Random Forest</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_te, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=['Pred: Tidak Putus', 'Pred: Putus'],
            y=['Aktual: Tidak Putus', 'Aktual: Putus'],
            colorscale=[[0,'#1a1f2e'],[0.5,'#3b82f6'],[1,'#ffd700']],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="<b>%{text}</b>", textfont=dict(size=18, color='white'),
        ))
        fig_cm.update_layout(title="Confusion Matrix",
                             paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f'Random Forest (AUC={roc_auc:.3f})',
                                     line=dict(color='#ffd700', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier',
                                     line=dict(color='#94a3b8', width=2, dash='dash')))
        fig_roc.update_layout(title=f"ROC Curve (AUC = {roc_auc:.3f})",
                              paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=400)
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("<div class='section-header'>🔍 Feature Importance (Top 10)</div>", unsafe_allow_html=True)
    fi_top = fi.head(10).copy()
    fi_top['feature_clean'] = (fi_top['feature']
                               .str.replace('_Enc', '', regex=False)
                               .str.replace('_', ' ', regex=False))
    fig_fi = go.Figure(go.Bar(
        x=fi_top['importance'], y=fi_top['feature_clean'],
        orientation='h',
        marker=dict(color=fi_top['importance'],
                    colorscale=[[0,'#3b82f6'],[1,'#ffd700']], showscale=False),
        text=[f"{v:.3f}" for v in fi_top['importance']],
        textposition='outside'
    ))
    fig_fi.update_layout(title="10 Fitur Paling Berpengaruh terhadap Putus Sekolah",
                         paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                         height=450, margin=dict(l=220))
    st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("📄 Lihat Classification Report Lengkap"):
        report = classification_report(y_te, y_pred, target_names=['Tidak Putus', 'Putus'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

# ══════ TAB 4 — PREDIKSI (DIPERBAIKI TOTAL) ════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-header'>🔮 Prediksi Risiko Putus Sekolah</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Isi formulir di bawah sesuai data kuesioner siswa untuk memprediksi risiko putus sekolah.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='background:rgba(255,215,0,0.07); border-left-color:#ffd700'>
    <b>ℹ️ Catatan Sistem Prediksi:</b> Prediksi menggunakan metode <b>hybrid</b> — menggabungkan
    model Random Forest (40%) dengan analisis risiko berbasis faktor domain (60%).
    Metode ini lebih akurat untuk kasus dengan profil risiko tinggi di berbagai dimensi.
    <br><br>
    <b>📌 Panduan pengisian Jam Kerja:</b> Dalam dataset, anak yang <b>putus sekolah</b>
    umumnya bekerja <b>15–39 jam/minggu</b>. Isi sesuai kondisi nyata anak.
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        # ── A. IDENTITAS ──────────────────────────────────────────────────────
        st.markdown("### 📋 A. Data Identitas Siswa")
        c1, c2, c3 = st.columns(3)
        with c1:
            usia = st.slider("Usia Anak (tahun)", 7, 18, 13)
        with c2:
            jk = st.selectbox("Jenis Kelamin", list(MAPPINGS['Jenis_Kelamin'].keys()))
        with c3:
            kelas = st.selectbox("Kelas Terakhir yang Ditempuh",
                                 list(MAPPINGS['Kelas_Terakhir'].keys()))
        st.markdown("---")

        # ── B. EKONOMI ────────────────────────────────────────────────────────
        st.markdown("### 💰 B. Faktor Ekonomi Keluarga")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            pendapatan = st.number_input(
                "Pendapatan Keluarga per Bulan (Rp)",
                min_value=300_000, max_value=5_000_000,
                value=800_000, step=50_000)
        with c5:
            tanggungan = st.slider("Jumlah Tanggungan Keluarga (orang)", 1, 12, 5)
        with c6:
            lahan = st.selectbox("Status Kepemilikan Lahan/Rumah",
                                 list(MAPPINGS['Kepemilikan_Lahan'].keys()))
        with c7:
            bansos = st.selectbox("Menerima Bantuan Sosial (Bansos)?",
                                  list(MAPPINGS['Status_Bansos'].keys()))

        c8, c9 = st.columns(2)
        with c8:
            pekerjaan_ayah = st.selectbox("Pekerjaan Ayah / Kepala Keluarga",
                                          list(MAPPINGS['Pekerjaan_Ayah'].keys()))
        with c9:
            pekerjaan_ibu = st.selectbox("Pekerjaan Ibu",
                                         list(MAPPINGS['Pekerjaan_Ibu'].keys()))

        c10, c11, c12 = st.columns(3)
        with c10:
            pend_ayah = st.selectbox("Pendidikan Terakhir Ayah",
                                     list(MAPPINGS['Pendidikan'].keys()))
        with c11:
            pend_ibu = st.selectbox("Pendidikan Terakhir Ibu",
                                    list(MAPPINGS['Pendidikan'].keys()))
        with c12:
            jam_kerja = st.slider(
                "Jam Kerja Anak per Minggu (jam)", 0, 50, 0,
                help="📌 Data lapangan: anak putus sekolah umumnya 15–39 jam/minggu. "
                     "Isi 0 jika anak tidak bekerja sama sekali.")
        st.markdown("---")

        # ── C. LINGKUNGAN ─────────────────────────────────────────────────────
        st.markdown("### 🌿 C. Faktor Lingkungan Tempat Tinggal")
        c13, c14, c15, c16 = st.columns(4)
        with c13:
            kondisi_ling = st.selectbox("Kondisi Wilayah Tempat Tinggal",
                                        list(MAPPINGS['Kondisi_Lingkungan'].keys()))
        with c14:
            listrik = st.selectbox("Akses Listrik di Rumah",
                                   list(MAPPINGS['Akses_Listrik'].keys()))
        with c15:
            internet = st.selectbox("Akses Internet di Rumah / Sekitar",
                                    list(MAPPINGS['Akses_Internet'].keys()))
        with c16:
            fasilitas = st.selectbox("Ketersediaan Fasilitas Belajar",
                                     list(MAPPINGS['Ketersediaan_Fasilitas_Belajar'].keys()))
        st.markdown("---")

        # ── D. JALAN & JARAK ──────────────────────────────────────────────────
        st.markdown("### 🛣️ D. Akses Jalan & Jarak ke Sekolah")
        c17, c18, c19 = st.columns(3)
        with c17:
            jarak = st.slider("Jarak Rumah ke Sekolah (km)", 0.5, 60.0, 5.0, 0.5)
        with c18:
            waktu = st.slider("Waktu Tempuh ke Sekolah (menit)", 5, 180, 30)
        with c19:
            transportasi = st.selectbox("Ketersediaan Transportasi",
                                        list(MAPPINGS['Ketersediaan_Transportasi'].keys()))

        c20, c21 = st.columns(2)
        with c20:
            jenis_jalan = st.selectbox("Jenis/Kondisi Jalan Menuju Sekolah",
                                       list(MAPPINGS['Jenis_Jalan'].keys()))
        with c21:
            cond_hujan = st.selectbox("Kondisi Jalan saat Musim Hujan",
                                      list(MAPPINGS['Kondisi_Jalan_Saat_Hujan'].keys()))
        st.markdown("---")

        # ── E. PSIKOLOGI & SOSIAL ─────────────────────────────────────────────
        st.markdown("### 💡 E. Faktor Psikologi & Sosial Siswa")
        c22, c23 = st.columns(2)
        with c22:
            minat = st.selectbox("Minat Belajar Anak",
                                 list(MAPPINGS['Skala5'].keys()))
        with c23:
            motivasi = st.selectbox("Motivasi Melanjutkan Sekolah",
                                    list(MAPPINGS['Skala5'].keys()))

        c24, c25 = st.columns(2)
        with c24:
            dukungan = st.selectbox("Dukungan Orang Tua terhadap Pendidikan Anak",
                                    list(MAPPINGS['Dukungan_Orang_Tua'].keys()))
        with c25:
            teman = st.selectbox("Pengaruh Teman Sebaya terhadap Semangat Sekolah",
                                 list(MAPPINGS['Pengaruh_Teman_Sebaya'].keys()))

        st.markdown("---")
        submitted = st.form_submit_button("🔮 Prediksi Risiko Sekarang", use_container_width=True)

    # ── PROSES PREDIKSI ───────────────────────────────────────────────────────
    if submitted:
        # Encode input
        input_values = [
            usia,
            MAPPINGS['Jenis_Kelamin'][jk],
            MAPPINGS['Kelas_Terakhir'][kelas],
            pendapatan,
            tanggungan,
            MAPPINGS['Kepemilikan_Lahan'][lahan],
            MAPPINGS['Status_Bansos'][bansos],
            MAPPINGS['Kondisi_Lingkungan'][kondisi_ling],
            MAPPINGS['Akses_Listrik'][listrik],
            MAPPINGS['Akses_Internet'][internet],
            MAPPINGS['Ketersediaan_Fasilitas_Belajar'][fasilitas],
            MAPPINGS['Pekerjaan_Ayah'][pekerjaan_ayah],
            MAPPINGS['Pekerjaan_Ibu'][pekerjaan_ibu],
            MAPPINGS['Pendidikan'][pend_ayah],
            MAPPINGS['Pendidikan'][pend_ibu],
            jam_kerja,
            jarak,
            MAPPINGS['Jenis_Jalan'][jenis_jalan],
            waktu,
            MAPPINGS['Ketersediaan_Transportasi'][transportasi],
            MAPPINGS['Kondisi_Jalan_Saat_Hujan'][cond_hujan],
            MAPPINGS['Skala5'][minat],
            MAPPINGS['Dukungan_Orang_Tua'][dukungan],
            MAPPINGS['Skala5'][motivasi],
            MAPPINGS['Pengaruh_Teman_Sebaya'][teman],
        ]

        input_df = pd.DataFrame([input_values], columns=FEATURE_COLS)

        # Prediksi RF
        prob_rf = model.predict_proba(input_df)[0]
        prob_putus_rf   = prob_rf[1] * 100

        # Rule-based risk score
        rule_inp = {
            'pendapatan': pendapatan,
            'jam_kerja': jam_kerja,
            'minat': MAPPINGS['Skala5'][minat],
            'motivasi': MAPPINGS['Skala5'][motivasi],
            'dukungan': MAPPINGS['Dukungan_Orang_Tua'][dukungan],
            'teman': MAPPINGS['Pengaruh_Teman_Sebaya'][teman],
            'jarak': jarak,
            'transportasi': MAPPINGS['Ketersediaan_Transportasi'][transportasi],
            'kondisi_ling': MAPPINGS['Kondisi_Lingkungan'][kondisi_ling],
            'listrik': MAPPINGS['Akses_Listrik'][listrik],
            'internet': MAPPINGS['Akses_Internet'][internet],
            'fasilitas': MAPPINGS['Ketersediaan_Fasilitas_Belajar'][fasilitas],
            'tanggungan': tanggungan,
        }
        rule_score = hitung_rule_risk_score(rule_inp)

        # Hybrid probability
        prob_putus_hybrid = 0.4 * prob_putus_rf + 0.6 * rule_score
        prob_putus_hybrid = min(prob_putus_hybrid, 100.0)
        prob_tidak_hybrid = 100.0 - prob_putus_hybrid

        # Final prediction
        pred = 1 if prob_putus_hybrid >= 50 else 0

        # Level risiko
        if prob_putus_hybrid >= 70:
            level = "🔴 RISIKO TINGGI";   level_color = "#f87171"
        elif prob_putus_hybrid >= 40:
            level = "🟡 RISIKO SEDANG";   level_color = "#fbbf24"
        else:
            level = "🟢 RISIKO RENDAH";   level_color = "#4ade80"

        # Tampilkan peringatan jika jam kerja sangat rendah tapi faktor lain buruk
        if jam_kerja < 10 and rule_score >= 60:
            st.markdown(f"""
            <div class='warning-box'>
            ⚠️ <b>Catatan Analisis:</b> Jam kerja anak rendah ({jam_kerja} jam/minggu),
            namun <b>skor risiko berbasis faktor lain sangat tinggi ({rule_score:.0f}/100)</b>.
            Dalam data lapangan, semua anak yang putus sekolah bekerja ≥15 jam/minggu —
            kemungkinan kondisi anak ini baru menuju ke arah putus sekolah atau
            faktor-faktor risiko lain sangat mendominasi.
            Sistem hybrid mendeteksi risiko tinggi berdasarkan keseluruhan profil.
            </div>
            """, unsafe_allow_html=True)

        # ── Hasil utama ───────────────────────────────────────────────────────
        if pred == 1:
            st.markdown(f"""
            <div class='pred-result pred-danger'>
                <div class='pred-title'>⚠️ BERISIKO PUTUS SEKOLAH</div>
                <div style='font-size:1.1rem; margin-top:0.6rem'>
                    Probabilitas Putus Sekolah:
                    <b style='color:#f87171; font-size:1.4rem'> {prob_putus_hybrid:.1f}%</b>
                    &nbsp;|&nbsp; Level: <b style='color:{level_color}'>{level}</b>
                </div>
                <div style='font-size:0.85rem; margin-top:0.8rem; color:#94a3b8'>
                    📌 Segera lakukan intervensi: beasiswa, bimbingan belajar, dan konseling motivasi.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='pred-result pred-safe'>
                <div class='pred-title'>✅ TIDAK BERISIKO PUTUS SEKOLAH</div>
                <div style='font-size:1.1rem; margin-top:0.6rem'>
                    Probabilitas Melanjutkan Sekolah:
                    <b style='color:#4ade80; font-size:1.4rem'> {prob_tidak_hybrid:.1f}%</b>
                    &nbsp;|&nbsp; Level: <b style='color:{level_color}'>{level}</b>
                </div>
                <div style='font-size:0.85rem; margin-top:0.8rem; color:#94a3b8'>
                    📌 Kondisi aman. Tetap pantau perkembangan secara berkala.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Detail skor komponen ──────────────────────────────────────────────
        col_detail1, col_detail2 = st.columns(2)
        with col_detail1:
            st.markdown(f"""
            <div class='info-box' style='background:rgba(59,130,246,0.08); border-left-color:#60a5fa'>
                <b style='color:#60a5fa'>🤖 Probabilitas Model RF:</b>
                <span style='float:right; font-weight:700'>{prob_putus_rf:.1f}%</span><br>
                <b style='color:#fbbf24'>📊 Skor Risiko Domain:</b>
                <span style='float:right; font-weight:700'>{rule_score:.1f}/100</span><br>
                <hr style='border-color:rgba(255,255,255,0.1); margin:0.5rem 0'>
                <b style='color:#ffd700'>🔀 Probabilitas Hybrid (Final):</b>
                <span style='float:right; font-weight:700; color:#ffd700'>{prob_putus_hybrid:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        with col_detail2:
            st.markdown(f"""
            <div class='info-box' style='background:rgba(255,215,0,0.05); border-left-color:#ffd700'>
                <b style='color:#ffd700'>📐 Formula Hybrid:</b><br>
                <span style='font-size:0.8rem; color:#94a3b8'>
                (40% × Prob RF) + (60% × Skor Domain)<br>
                = (40% × {prob_putus_rf:.1f}%) + (60% × {rule_score:.1f}%)<br>
                = <b style='color:#ffd700'>{prob_putus_hybrid:.1f}%</b>
                </span>
            </div>
            """, unsafe_allow_html=True)

        # ── Gauge Chart ───────────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_putus_hybrid,
            delta={'reference': 50, 'increasing': {'color': RED}, 'decreasing': {'color': GREEN}},
            title={'text': "Probabilitas Putus Sekolah — Hybrid (%)", 'font': {'size': 14, 'color': 'white'}},
            number={'suffix': '%', 'font': {'size': 40, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white',
                         'tickfont': {'color': 'white'}, 'ticksuffix': '%'},
                'bar': {'color': RED if prob_putus_hybrid > 50 else GREEN},
                'steps': [
                    {'range': [0,  40], 'color': 'rgba(34,197,94,0.25)'},
                    {'range': [40, 70], 'color': 'rgba(245,158,11,0.25)'},
                    {'range': [70,100], 'color': 'rgba(220,38,38,0.25)'},
                ],
                'threshold': {
                    'line': {'color': '#ffd700', 'width': 4},
                    'thickness': 0.75, 'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', height=320,
            margin=dict(t=60, b=20), font=dict(color='white'))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Analisis Faktor Risiko ─────────────────────────────────────────────
        st.markdown("<div class='section-header'>🔍 Analisis Faktor Risiko dari Data Input</div>",
                    unsafe_allow_html=True)

        risk_factors = []
        safe_factors = []

        if pendapatan < 1_000_000:
            risk_factors.append(f"💸 Pendapatan keluarga sangat rendah (Rp {pendapatan:,.0f}/bln)")
        else:
            safe_factors.append(f"💰 Pendapatan keluarga cukup (Rp {pendapatan:,.0f}/bln)")

        if jarak > 15:
            risk_factors.append(f"📍 Jarak ke sekolah jauh ({jarak} km)")
        else:
            safe_factors.append(f"📍 Jarak ke sekolah terjangkau ({jarak} km)")

        if MAPPINGS['Skala5'][minat] <= 1:
            risk_factors.append(f"📚 Minat belajar anak rendah ({minat})")
        else:
            safe_factors.append(f"📚 Minat belajar anak baik ({minat})")

        if MAPPINGS['Dukungan_Orang_Tua'][dukungan] <= 1:
            risk_factors.append(f"👨‍👩‍👧 Dukungan orang tua kurang ({dukungan})")
        else:
            safe_factors.append(f"👨‍👩‍👧 Dukungan orang tua memadai ({dukungan})")

        if jam_kerja >= 15:
            risk_factors.append(f"⏱️ Jam kerja anak tinggi ({jam_kerja} jam/minggu) — faktor risiko utama")
        elif jam_kerja > 0:
            risk_factors.append(f"⏱️ Anak bekerja {jam_kerja} jam/minggu (perlu dipantau)")
        else:
            safe_factors.append("⏱️ Anak tidak dibebani pekerjaan")

        if kondisi_ling in ["Sangat Terpencil", "Terpencil"]:
            risk_factors.append(f"🏚️ Wilayah tinggal {kondisi_ling.lower()}")
        else:
            safe_factors.append(f"🏡 Wilayah tinggal {kondisi_ling.lower()}")

        if MAPPINGS['Ketersediaan_Fasilitas_Belajar'][fasilitas] <= 1:
            risk_factors.append(f"🖊️ Fasilitas belajar {fasilitas.lower()}")
        else:
            safe_factors.append(f"🖊️ Fasilitas belajar {fasilitas.lower()}")

        if transportasi == "Tidak Ada":
            risk_factors.append("🚌 Tidak ada akses transportasi ke sekolah")
        else:
            safe_factors.append(f"🚌 Transportasi tersedia ({transportasi})")

        if internet == "Tidak":
            risk_factors.append("📶 Tidak ada akses internet")
        else:
            safe_factors.append("📶 Memiliki akses internet")

        if tanggungan >= 6:
            risk_factors.append(f"👪 Tanggungan keluarga banyak ({tanggungan} orang)")
        else:
            safe_factors.append(f"👪 Jumlah tanggungan wajar ({tanggungan} orang)")

        col_r, col_s = st.columns(2)
        with col_r:
            st.markdown(f"""
            <div class='info-box' style='border-left-color:#f87171; background:rgba(220,38,38,0.08)'>
                <b style='color:#f87171'>⚠️ Faktor Risiko yang Ditemukan ({len(risk_factors)})</b><br><br>
                {'<br>'.join([f'• {r}' for r in risk_factors]) if risk_factors else '• Tidak ada faktor risiko signifikan'}
            </div>""", unsafe_allow_html=True)
        with col_s:
            st.markdown(f"""
            <div class='info-box' style='border-left-color:#4ade80; background:rgba(34,197,94,0.08)'>
                <b style='color:#4ade80'>✅ Faktor Pelindung yang Ditemukan ({len(safe_factors)})</b><br><br>
                {'<br>'.join([f'• {s}' for s in safe_factors]) if safe_factors else '• Tidak ada faktor pelindung yang teridentifikasi'}
            </div>""", unsafe_allow_html=True)

        # ── Rekomendasi Intervensi ────────────────────────────────────────────
        if pred == 1 or prob_putus_hybrid >= 40:
            st.markdown("<div class='section-header'>📋 Rekomendasi Intervensi</div>",
                        unsafe_allow_html=True)
            recs = []
            if pendapatan < 1_000_000:
                recs.append(("🔴 PRIORITAS TINGGI", "Ekonomi",
                              "Daftarkan keluarga ke program beasiswa penuh (KIP) dan bantuan sosial PKH/BPNT"))
            if jarak > 15 or transportasi == "Tidak Ada":
                recs.append(("🔴 PRIORITAS TINGGI", "Transportasi",
                              "Sediakan layanan antar-jemput gratis atau bangun asrama dekat sekolah"))
            if MAPPINGS['Skala5'][minat] <= 1:
                recs.append(("🟡 PRIORITAS SEDANG", "Motivasi",
                              "Lakukan program mentoring, ekstrakurikuler menarik, dan konseling siswa"))
            if MAPPINGS['Dukungan_Orang_Tua'][dukungan] <= 1:
                recs.append(("🟡 PRIORITAS SEDANG", "Orang Tua",
                              "Adakan penyuluhan orang tua tentang pentingnya pendidikan dan dampak jangka panjang"))
            if jam_kerja >= 15:
                recs.append(("🔴 PRIORITAS TINGGI", "Beban Kerja Anak",
                              f"Kurangi jam kerja anak dari {jam_kerja} jam/minggu — koordinasikan dengan keluarga & pemerintah desa"))
            elif jam_kerja > 5:
                recs.append(("🟡 PRIORITAS SEDANG", "Beban Kerja Anak",
                              "Koordinasi dengan keluarga untuk mengurangi jam kerja anak"))
            if MAPPINGS['Ketersediaan_Fasilitas_Belajar'][fasilitas] <= 1:
                recs.append(("🟢 PRIORITAS RENDAH", "Fasilitas",
                              "Bangun pojok baca / taman belajar masyarakat di lingkungan desa"))

            for prioritas, bidang, saran in recs:
                color = "#f87171" if "TINGGI" in prioritas else ("#fbbf24" if "SEDANG" in prioritas else "#4ade80")
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03); border-radius:10px; padding:0.8rem 1rem;
                            margin-bottom:0.6rem; border-left:4px solid {color}'>
                    <span style='font-size:0.75rem; font-weight:700; color:{color}'>{prioritas}</span>
                    &nbsp;·&nbsp; <b style='color:#e2e8f0'>{bidang}</b><br>
                    <span style='font-size:0.85rem; color:#94a3b8'>{saran}</span>
                </div>""", unsafe_allow_html=True)

# ══════ TAB 5 ══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-header'>📋 Dataset Lengkap</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Data", total)
    with col2: st.metric("Putus Sekolah", putus)
    with col3: st.metric("Tidak Putus", tidak)
    st.dataframe(df.drop(columns=['Label']).head(50), use_container_width=True)
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download CSV Lengkap", csv_data,
        "dataset_putus_sekolah_2026.csv", "text/csv",
        use_container_width=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="group-name">👩‍🎓 Kelompok Penelitian Data Science</div>
    <div style="margin:0.4rem 0">
        Gladis Primadona (2024020179) &nbsp;·&nbsp;
        Aulia Virgara (2024020230) &nbsp;·&nbsp;
        Jesika Tarigan (2024020119)
    </div>
    <div>Analisis Penentu Tingkat Putus Sekolah di Daerah Pedesaan · Random Forest Hybrid · 2026</div>
</div>
""", unsafe_allow_html=True)
