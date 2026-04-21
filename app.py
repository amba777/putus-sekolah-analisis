import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Putus Sekolah Pedesaan 2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DARK MODE PREMIUM CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1422 50%, #13182a 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0e1a 100%) !important;
    border-right: 1px solid rgba(255,215,0,0.2);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSlider label {
    color: #ffd700 !important;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.main-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1422 50%, #1a1f2e 100%);
    padding: 2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,215,0,0.15);
}

.main-title {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #ffd700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.main-subtitle {
    font-size: 0.9rem;
    color: #94a3b8 !important;
}

.metric-card {
    background: rgba(18, 22, 35, 0.9);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid rgba(255,215,0,0.15);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(255,215,0,0.4);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
}

.metric-badge {
    display: inline-block;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-top: 0.5rem;
}

.badge-danger { background: rgba(220,38,38,0.2); color: #f87171; }
.badge-success { background: rgba(34,197,94,0.2); color: #4ade80; }
.badge-info { background: rgba(59,130,246,0.2); color: #60a5fa; }
.badge-warning { background: rgba(245,158,11,0.2); color: #fbbf24; }

.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    border-left: 4px solid #ffd700;
    padding-left: 1rem;
    margin: 1.5rem 0 1rem 0;
}

.section-sub {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1.2rem;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-radius: 12px 12px 0 0;
    padding: 0.3rem 0.3rem 0;
    gap: 4px;
    border-bottom: 1px solid rgba(255,215,0,0.2);
}

.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.6rem 1.2rem;
    border-radius: 8px 8px 0 0;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    color: #ffd700 !important;
    background: rgba(255,215,0,0.1) !important;
    border-bottom: 2px solid #ffd700 !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background: rgba(18, 22, 35, 0.6);
    backdrop-filter: blur(5px);
    border-radius: 0 12px 12px 12px;
    padding: 1.5rem;
    border: 1px solid rgba(255,215,0,0.1);
}

.info-box {
    background: rgba(59,130,246,0.1);
    border-left: 4px solid #3b82f6;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin: 1rem 0;
    font-size: 0.85rem;
    color: #cbd5e1;
}

.pred-result {
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    margin: 1rem 0;
}

.pred-danger {
    background: linear-gradient(135deg, rgba(220,38,38,0.2), rgba(220,38,38,0.1));
    border: 2px solid #f87171;
}

.pred-safe {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.3);
}

.pred-title {
    font-size: 1.5rem;
    font-weight: 700;
}

.pred-danger .pred-title { color: #f87171; }
.pred-safe .pred-title { color: #4ade80; }

.stButton > button {
    background: linear-gradient(135deg, #ffd700, #b8860b);
    color: #0a0e1a !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 700;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #ffed4a, #daa520);
    transform: translateY(-2px);
}

.stRadio > div {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.stRadio > div label {
    background: rgba(255,255,255,0.05);
    padding: 0.4rem 1rem;
    border-radius: 30px;
    font-size: 0.8rem;
    color: #cbd5e1;
    border: 1px solid rgba(255,215,0,0.2);
}

.footer {
    background: linear-gradient(135deg, #0d1117, #0a0e1a);
    color: #94a3b8;
    padding: 1.5rem;
    border-radius: 16px;
    margin-top: 2rem;
    text-align: center;
    font-size: 0.75rem;
    border-top: 1px solid rgba(255,215,0,0.2);
}

.footer .group-name {
    color: #ffd700;
    font-weight: 600;
    font-size: 0.85rem;
}

.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #ffd700, transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_putus_sekolah.csv")

@st.cache_data
def prepare_model_data(df):
    df_enc = df.copy()
    
    # Buat mapping manual untuk memastikan konsistensi
    mappings = {}
    
    # Mapping untuk setiap kolom kategorikal
    # Kondisi Lingkungan
    kondisi_map = {"Sangat Terpencil": 0, "Terpencil": 1, "Cukup Terjangkau": 2, "Terjangkau": 3}
    df_enc['Kondisi_Lingkungan_Enc'] = df_enc['Kondisi_Lingkungan'].map(kondisi_map)
    mappings['Kondisi_Lingkungan'] = kondisi_map
    
    # Akses Listrik
    listrik_map = {"Tidak": 0, "Ya": 1}
    df_enc['Akses_Listrik_Enc'] = df_enc['Akses_Listrik'].map(listrik_map)
    mappings['Akses_Listrik'] = listrik_map
    
    # Akses Internet
    internet_map = {"Tidak": 0, "Ya": 1}
    df_enc['Akses_Internet_Enc'] = df_enc['Akses_Internet'].map(internet_map)
    mappings['Akses_Internet'] = internet_map
    
    # Fasilitas Belajar
    fasilitas_map = {"Sangat Kurang": 0, "Kurang": 1, "Cukup": 2, "Baik": 3}
    df_enc['Ketersediaan_Fasilitas_Belajar_Enc'] = df_enc['Ketersediaan_Fasilitas_Belajar'].map(fasilitas_map)
    mappings['Ketersediaan_Fasilitas_Belajar'] = fasilitas_map
    
    # Kepemilikan Lahan
    lahan_map = {"Tidak Punya": 0, "Sewa": 1, "Milik Sendiri": 2}
    df_enc['Kepemilikan_Lahan_Enc'] = df_enc['Kepemilikan_Lahan'].map(lahan_map)
    mappings['Kepemilikan_Lahan'] = lahan_map
    
    # Status Bansos
    bansos_map = {"Tidak": 0, "Ya": 1}
    df_enc['Status_Bansos_Enc'] = df_enc['Status_Bansos'].map(bansos_map)
    mappings['Status_Bansos'] = bansos_map
    
    # Jenis Kelamin
    gender_map = {"Perempuan": 0, "Laki-laki": 1}
    df_enc['Jenis_Kelamin_Enc'] = df_enc['Jenis_Kelamin'].map(gender_map)
    mappings['Jenis_Kelamin'] = gender_map
    
    # Pekerjaan Ayah
    pekerjaan_ayah_map = {"Ibu Rumah Tangga": 0, "Tidak Bekerja": 1, "Buruh": 2, "Buruh Harian": 3, 
                          "Nelayan": 4, "Pedagang Kecil": 5, "Pekerja Serabutan": 6, "Petani": 7}
    df_enc['Pekerjaan_Ayah_Enc'] = df_enc['Pekerjaan_Ayah'].map(pekerjaan_ayah_map)
    mappings['Pekerjaan_Ayah'] = pekerjaan_ayah_map
    
    # Pekerjaan Ibu
    pekerjaan_ibu_map = {"Tidak Bekerja": 0, "Buruh": 1, "Ibu Rumah Tangga": 2, "Pedagang Kecil": 3, "Petani": 4}
    df_enc['Pekerjaan_Ibu_Enc'] = df_enc['Pekerjaan_Ibu'].map(pekerjaan_ibu_map)
    mappings['Pekerjaan_Ibu'] = pekerjaan_ibu_map
    
    # Pendidikan
    pendidikan_map = {"Tidak Sekolah": 0, "SD": 1, "SMP": 2, "SMA": 3, "Diploma/S1": 4}
    df_enc['Pendidikan_Ayah_Enc'] = df_enc['Pendidikan_Ayah'].map(pendidikan_map)
    df_enc['Pendidikan_Ibu_Enc'] = df_enc['Pendidikan_Ibu'].map(pendidikan_map)
    mappings['Pendidikan'] = pendidikan_map
    
    # Jenis Jalan
    jalan_map = {"Jalan Tanah": 0, "Jalan Kerikil": 1, "Jalan Aspal Rusak": 2, "Jalan Aspal Baik": 3}
    df_enc['Jenis_Jalan_Enc'] = df_enc['Jenis_Jalan'].map(jalan_map)
    mappings['Jenis_Jalan'] = jalan_map
    
    # Transportasi
    transportasi_map = {"Tidak Ada": 0, "Ojek/Becak": 1, "Angkutan Umum": 2, "Kendaraan Pribadi": 3}
    df_enc['Ketersediaan_Transportasi_Enc'] = df_enc['Ketersediaan_Transportasi'].map(transportasi_map)
    mappings['Ketersediaan_Transportasi'] = transportasi_map
    
    # Kondisi Jalan Hujan
    hujan_map = {"Tidak Bisa Dilalui": 0, "Sangat Sulit": 1, "Sulit": 2, "Bisa Dilalui": 3}
    df_enc['Kondisi_Jalan_Saat_Hujan_Enc'] = df_enc['Kondisi_Jalan_Saat_Hujan'].map(hujan_map)
    mappings['Kondisi_Jalan_Saat_Hujan'] = hujan_map
    
    # Minat, Dukungan, Motivasi, Teman
    minat_map = {"Sangat Rendah": 0, "Rendah": 1, "Sedang": 2, "Tinggi": 3, "Sangat Tinggi": 4}
    df_enc['Minat_Belajar_Anak_Enc'] = df_enc['Minat_Belajar_Anak'].map(minat_map)
    df_enc['Dukungan_Orang_Tua_Enc'] = df_enc['Dukungan_Orang_Tua'].map(minat_map)
    df_enc['Motivasi_Melanjutkan_Sekolah_Enc'] = df_enc['Motivasi_Melanjutkan_Sekolah'].map(minat_map)
    df_enc['Pengaruh_Teman_Sebaya_Enc'] = df_enc['Pengaruh_Teman_Sebaya'].map(minat_map)
    mappings['Minat'] = minat_map
    
    # Kelas Terakhir
    kelas_list = ["SD Kelas 1","SD Kelas 2","SD Kelas 3","SD Kelas 4","SD Kelas 5","SD Kelas 6",
                  "SMP Kelas 7","SMP Kelas 8","SMP Kelas 9","SMA Kelas 10","SMA Kelas 11","SMA Kelas 12"]
    kelas_map = {k: i for i, k in enumerate(kelas_list)}
    df_enc['Kelas_Terakhir_Enc'] = df_enc['Kelas_Terakhir'].map(kelas_map)
    mappings['Kelas_Terakhir'] = kelas_map
    
    # Feature columns (numerik + encoded)
    feature_cols = [
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
    
    X = df_enc[feature_cols]
    y = df_enc['Label']
    
    return X, y, feature_cols, mappings

df = load_data()
X, y, feature_cols, mappings = prepare_model_data(df)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.5rem 0 0.5rem'>
        <div style='font-size:2.5rem'>🎓</div>
        <div style='font-size:1rem; font-weight:700; color:#ffd700; margin-top:0.3rem'>
            Analisis Putus Sekolah
        </div>
        <div style='font-size:0.7rem; color:#94a3b8'>Data Mining Dashboard • 2026</div>
        <div style='height:1px; background:linear-gradient(90deg,transparent,#ffd700,transparent); margin:1rem 0'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.7rem; color:#ffd700; font-weight:600; letter-spacing:0.1em'>⚙️ MODEL CONFIGURATION</p>", unsafe_allow_html=True)

    st.info("📌 **Metode yang digunakan:** Random Forest\n\nAkurasi lebih tinggi & tahan overfitting")
    
    test_size = st.slider("Test Split (%)", 20, 40, 30, 5)
    max_depth = st.slider("Max Depth", 2, 10, 5)
    n_trees = st.slider("Jumlah Pohon (n_estimators)", 50, 200, 100, 25)

    st.markdown("<div style='height:1px; background:linear-gradient(90deg,transparent,#ffd700,transparent); margin:1rem 0'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem; color:#ffd700; font-weight:600; letter-spacing:0.1em'>📋 KELOMPOK PENELITI</p>", unsafe_allow_html=True)

    members = [
        ("👩‍🎓", "Gladis Primadona", "2024020179"),
        ("👩‍🎓", "Aulia Virgara", "2024020230"),
        ("👩‍🎓", "Jesika Tarigan", "2024020119"),
    ]
    for icon, name, nirm in members:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03); border-radius:10px; padding:0.5rem 0.7rem; margin-bottom:0.5rem; border-left:2px solid #ffd700'>
            <div style='font-size:0.8rem; font-weight:600; color:#ffd700'>{icon} {name}</div>
            <div style='font-size:0.65rem; color:#94a3b8'>NIRM: {nirm}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Train Model ───────────────────────────────────────────────────────────────
@st.cache_data
def train_model(test_sz, depth, n_est):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_sz/100,
                                                random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                    random_state=42, class_weight='balanced')
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    cv = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    fi = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    return model, X_te, y_te, y_pred, y_prob, acc, cv, fi

model, X_te, y_te, y_pred, y_prob, acc, cv, fi = train_model(test_size, max_depth, n_trees)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">📊 Analisis Penentu Tingkat Putus Sekolah</div>
    <div class="main-subtitle">Di Daerah Pedesaan · Random Forest Classification · Studi Kasus 100 Responden · 2026</div>
</div>
""", unsafe_allow_html=True)

# ─── KPI Cards ─────────────────────────────────────────────────────────────────
total = len(df)
putus = (df['Status_Putus_Sekolah'] == 'Ya').sum()
tidak = (df['Status_Putus_Sekolah'] == 'Tidak').sum()
pct_putus = putus / total * 100

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, str(total), "Total Responden", "badge-info", "📁 Dataset 2026"),
    (c2, str(putus), "Putus Sekolah", "badge-danger", f"⚠️ {pct_putus:.1f}% dari total"),
    (c3, str(tidak), "Tidak Putus", "badge-success", f"✅ {100-pct_putus:.1f}% dari total"),
    (c4, f"{acc*100:.1f}%", "Akurasi Model", "badge-info", f"🤖 Random Forest"),
    (c5, f"{cv*100:.1f}%", "CV Akurasi", "badge-warning", "🔁 5-Fold Cross Val"),
]
for col, val, lbl, badge, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
            <span class="metric-badge {badge}">{sub}</span>
        </div>
        """, unsafe_allow_html=True)

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

RED = "#f87171"
GREEN = "#4ade80"

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 – TENTANG METODE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div class='section-header'>🌲 Tentang Random Forest</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='info-box' style='background:rgba(255,215,0,0.05); border-left-color:#ffd700;'>
        <b>📌 Apa itu Random Forest?</b><br>
        Random Forest adalah algoritma <b>ensemble learning</b> yang menggabungkan banyak <b>Decision Tree</b> 
        untuk menghasilkan prediksi yang lebih akurat dan stabil.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box' style='background:rgba(59,130,246,0.05); border-left-color:#3b82f6; margin-top:1rem'>
        <b>⚙️ Cara Kerja Random Forest:</b><br>
        1. Membuat <b>N pohon keputusan</b> (n_estimators) dari sampel data acak<br>
        2. Setiap pohon memilih <b>fitur secara acak</b> untuk menentukan split terbaik<br>
        3. Setiap pohon memberikan <b>prediksi</b> (klasifikasi/regresi)<br>
        4. Hasil akhir = <b>voting mayoritas</b> dari semua pohon<br>
        5. Feature importance dihitung dari rata-rata penurunan impurity
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background:rgba(255,215,0,0.1); border-radius:16px; padding:1rem; text-align:center; border:1px solid rgba(255,215,0,0.3)'>
            <div style='font-size:3rem'>🌲🌲🌲</div>
            <div style='font-size:1.2rem; font-weight:700; color:#ffd700; margin:0.5rem 0'>Random Forest</div>
            <hr style='border-color:rgba(255,215,0,0.2)'>
            <div style='text-align:left; font-size:0.75rem; color:#cbd5e1'>
            ✅ Akurasi tinggi<br>
            ✅ Tahan overfitting<br>
            ✅ Feature importance<br>
            ✅ Handle data kategorikal
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – EKSPLORASI DATA (disingkat karena panjang)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-header'>📊 Distribusi & Gambaran Umum Data</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_pie = go.Figure(go.Pie(
            labels=['Putus Sekolah', 'Tidak Putus'],
            values=[putus, tidak],
            hole=0.55,
            marker_colors=[RED, GREEN],
            textfont=dict(size=13, color='white'),
        ))
        fig_pie.add_annotation(text=f"<b>{pct_putus:.0f}%</b><br>Putus", x=0.5, y=0.5,
                                font=dict(size=14, color='white'), showarrow=False)
        fig_pie.update_layout(title="Distribusi Status Putus Sekolah", paper_bgcolor='rgba(0,0,0,0)', 
                              font=dict(color='white'), height=340)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_age = px.histogram(df, x='Usia_Anak', color='Status_Putus_Sekolah',
                                color_discrete_map={'Ya': RED, 'Tidak': GREEN},
                                nbins=11, barmode='overlay', opacity=0.8)
        fig_age.update_layout(title="Distribusi Usia Responden", paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='white'), height=340)
        st.plotly_chart(fig_age, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – VISUALISASI FAKTOR (disingkat)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-header'>📈 Analisis Per Faktor Penentu</div>", unsafe_allow_html=True)
    st.info("Silakan pilih tab faktor di atas untuk melihat visualisasi masing-masing faktor.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – HASIL MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-header'>🤖 Hasil Model: Random Forest</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_te, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=['Pred: Tidak Putus', 'Pred: Putus'],
            y=['Aktual: Tidak Putus', 'Aktual: Putus'],
            colorscale=[[0,'#1a1f2e'],[0.5,'#3b82f6'],[1,'#ffd700']],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="<b>%{text}</b>", textfont=dict(size=18, color='white'),
        ))
        fig_cm.update_layout(title="Confusion Matrix", paper_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Random Forest (AUC = {roc_auc:.3f})',
                                      line=dict(color='#ffd700', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier',
                                      line=dict(color='#94a3b8', width=2, dash='dash')))
        fig_roc.update_layout(title=f"ROC Curve (AUC = {roc_auc:.3f})", paper_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("<div class='section-header'>🔍 Feature Importance (Top 10)</div>", unsafe_allow_html=True)
    fi_top = fi.head(10).copy()
    fi_top['feature_clean'] = fi_top['feature'].str.replace('_Enc', '').str.replace('_', ' ')

    fig_fi = go.Figure(go.Bar(
        x=fi_top['importance'], y=fi_top['feature_clean'],
        orientation='h',
        marker=dict(color=fi_top['importance'], colorscale=[[0,'#3b82f6'],[1,'#ffd700']], showscale=False),
        text=[f"{v:.3f}" for v in fi_top['importance']],
        textposition='outside'
    ))
    fig_fi.update_layout(title="10 Fitur Paling Berpengaruh", paper_bgcolor='rgba(0,0,0,0)', height=450, margin=dict(l=200))
    st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PREDIKSI (DIPERBAIKI TOTAL)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-header'>🔮 Prediksi Individual</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Masukkan data untuk memprediksi risiko putus sekolah.</div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("**📋 Data Identitas**")
        c1, c2, c3 = st.columns(3)
        with c1:
            usia = st.slider("Usia Anak (tahun)", 7, 18, 14)
        with c2:
            jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        with c3:
            kelas = st.selectbox("Kelas Terakhir", ["SD Kelas 1","SD Kelas 2","SD Kelas 3","SD Kelas 4","SD Kelas 5","SD Kelas 6","SMP Kelas 7","SMP Kelas 8","SMP Kelas 9","SMA Kelas 10","SMA Kelas 11","SMA Kelas 12"])

        st.markdown("**💰 Faktor Ekonomi**")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            pendapatan = st.number_input("Pendapatan Keluarga/Bulan (Rp)", 300000, 5000000, 300000, step=50000)
        with c5:
            tanggungan = st.slider("Jumlah Tanggungan Keluarga", 1, 10, 8)
        with c6:
            lahan = st.selectbox("Kepemilikan Lahan", ["Tidak Punya", "Sewa", "Milik Sendiri"])
        with c7:
            bansos = st.selectbox("Penerima Bansos", ["Ya", "Tidak"])

        st.markdown("**🌿 Faktor Lingkungan**")
        c8, c9, c10, c11 = st.columns(4)
        with c8:
            kondisi_ling = st.selectbox("Kondisi Lingkungan", ["Sangat Terpencil", "Terpencil", "Cukup Terjangkau", "Terjangkau"])
        with c9:
            listrik = st.selectbox("Akses Listrik", ["Ya", "Tidak"])
        with c10:
            internet = st.selectbox("Akses Internet", ["Ya", "Tidak"])
        with c11:
            fasilitas = st.selectbox("Fasilitas Belajar", ["Sangat Kurang", "Kurang", "Cukup", "Baik"])

        st.markdown("**👨‍👩‍👧 Pekerjaan Orang Tua**")
        c12, c13, c14, c15, c16 = st.columns(5)
        with c12:
            pekerjaan_ayah = st.selectbox("Pekerjaan Ayah", ["Petani", "Buruh Harian", "Nelayan", "Pedagang Kecil", "Pekerja Serabutan", "Tidak Bekerja"])
        with c13:
            pekerjaan_ibu = st.selectbox("Pekerjaan Ibu", ["Ibu Rumah Tangga", "Petani", "Buruh", "Pedagang Kecil", "Tidak Bekerja"])
        with c14:
            pend_ayah = st.selectbox("Pendidikan Ayah", ["Tidak Sekolah", "SD", "SMP", "SMA", "Diploma/S1"])
        with c15:
            pend_ibu = st.selectbox("Pendidikan Ibu", ["Tidak Sekolah", "SD", "SMP", "SMA", "Diploma/S1"])
        with c16:
            jam_kerja = st.slider("Jam Kerja Anak per Minggu (jam)", 0, 50, 30)

        st.markdown("**🛣️ Jalan & Jarak**")
        c17, c18, c19, c20, c21 = st.columns(5)
        with c17:
            jarak = st.slider("Jarak ke Sekolah (km)", 1.0, 60.0, 25.0, 0.5)
        with c18:
            jenis_jalan = st.selectbox("Jenis Jalan", ["Jalan Tanah", "Jalan Kerikil", "Jalan Aspal Rusak", "Jalan Aspal Baik"])
        with c19:
            waktu = st.slider("Waktu Tempuh (menit)", 5, 180, 90)
        with c20:
            transportasi = st.selectbox("Ketersediaan Transportasi", ["Tidak Ada", "Ojek/Becak", "Angkutan Umum", "Kendaraan Pribadi"])
        with c21:
            cond_hujan = st.selectbox("Kondisi Jalan Saat Hujan", ["Tidak Bisa Dilalui", "Sangat Sulit", "Sulit", "Bisa Dilalui"])

        st.markdown("**💡 Minat & Motivasi**")
        c22, c23, c24, c25 = st.columns(4)
        with c22:
            minat = st.selectbox("Minat Belajar Anak", ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"])
        with c23:
            dukungan = st.selectbox("Dukungan Orang Tua", ["Sangat Kurang", "Kurang", "Cukup", "Baik", "Sangat Baik"])
        with c24:
            motivasi = st.selectbox("Motivasi Melanjutkan Sekolah", ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"])
        with c25:
            teman = st.selectbox("Pengaruh Teman Sebaya", ["Sangat Negatif", "Negatif", "Netral", "Positif", "Sangat Positif"])

        submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

    if submitted:
        # ENCODE semua input menggunakan mapping yang sama
        input_encoded = []
        
        # Usia
        input_encoded.append(usia)
        
        # Jenis Kelamin
        gender_val = mappings['Jenis_Kelamin'].get(jk, 0)
        input_encoded.append(gender_val)
        
        # Kelas
        kelas_val = mappings['Kelas_Terakhir'].get(kelas, 0)
        input_encoded.append(kelas_val)
        
        # Pendapatan
        input_encoded.append(pendapatan)
        
        # Tanggungan
        input_encoded.append(tanggungan)
        
        # Kepemilikan Lahan
        lahan_val = mappings['Kepemilikan_Lahan'].get(lahan, 0)
        input_encoded.append(lahan_val)
        
        # Status Bansos
        bansos_val = mappings['Status_Bansos'].get(bansos, 0)
        input_encoded.append(bansos_val)
        
        # Kondisi Lingkungan
        kondisi_val = mappings['Kondisi_Lingkungan'].get(kondisi_ling, 0)
        input_encoded.append(kondisi_val)
        
        # Akses Listrik
        listrik_val = mappings['Akses_Listrik'].get(listrik, 0)
        input_encoded.append(listrik_val)
        
        # Akses Internet
        internet_val = mappings['Akses_Internet'].get(internet, 0)
        input_encoded.append(internet_val)
        
        # Fasilitas Belajar
        fasilitas_val = mappings['Ketersediaan_Fasilitas_Belajar'].get(fasilitas, 0)
        input_encoded.append(fasilitas_val)
        
        # Pekerjaan Ayah
        pekerjaan_ayah_val = mappings['Pekerjaan_Ayah'].get(pekerjaan_ayah, 0)
        input_encoded.append(pekerjaan_ayah_val)
        
        # Pekerjaan Ibu
        pekerjaan_ibu_val = mappings['Pekerjaan_Ibu'].get(pekerjaan_ibu, 0)
        input_encoded.append(pekerjaan_ibu_val)
        
        # Pendidikan Ayah
        pend_ayah_val = mappings['Pendidikan'].get(pend_ayah, 0)
        input_encoded.append(pend_ayah_val)
        
        # Pendidikan Ibu
        pend_ibu_val = mappings['Pendidikan'].get(pend_ibu, 0)
        input_encoded.append(pend_ibu_val)
        
        # Jam Kerja
        input_encoded.append(jam_kerja)
        
        # Jarak
        input_encoded.append(jarak)
        
        # Jenis Jalan
        jalan_val = mappings['Jenis_Jalan'].get(jenis_jalan, 0)
        input_encoded.append(jalan_val)
        
        # Waktu Tempuh
        input_encoded.append(waktu)
        
        # Transportasi
        transportasi_val = mappings['Ketersediaan_Transportasi'].get(transportasi, 0)
        input_encoded.append(transportasi_val)
        
        # Kondisi Jalan Hujan
        hujan_val = mappings['Kondisi_Jalan_Saat_Hujan'].get(cond_hujan, 0)
        input_encoded.append(hujan_val)
        
        # Minat
        minat_val = mappings['Minat'].get(minat, 0)
        input_encoded.append(minat_val)
        
        # Dukungan
        dukungan_val = mappings['Minat'].get(dukungan, 0)
        input_encoded.append(dukungan_val)
        
        # Motivasi
        motivasi_val = mappings['Minat'].get(motivasi, 0)
        input_encoded.append(motivasi_val)
        
        # Pengaruh Teman
        teman_val = mappings['Minat'].get(teman, 0)
        input_encoded.append(teman_val)
        
        # Buat DataFrame
        input_df = pd.DataFrame([input_encoded], columns=feature_cols)
        
        # Prediksi
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        
        # Tampilkan hasil
        if pred == 1:
            st.markdown(f"""
            <div class='pred-result pred-danger'>
                <div class='pred-title'>⚠️ BERISIKO PUTUS SEKOLAH</div>
                <div style='font-size:1.2rem; margin-top:0.5rem'>
                    Probabilitas Putus Sekolah: <b style='color:#f87171'>{prob[1]*100:.1f}%</b>
                </div>
                <div style='font-size:0.9rem; margin-top:0.5rem; color:#94a3b8'>
                    📌 Rekomendasi: Segera lakukan intervensi berupa beasiswa, bimbingan belajar, dan konseling motivasi.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='pred-result pred-safe'>
                <div class='pred-title'>✅ AMAN — TIDAK BERISIKO</div>
                <div style='font-size:1.2rem; margin-top:0.5rem'>
                    Probabilitas Melanjutkan Sekolah: <b style='color:#4ade80'>{prob[0]*100:.1f}%</b>
                </div>
                <div style='font-size:0.9rem; margin-top:0.5rem; color:#94a3b8'>
                    📌 Rekomendasi: Kondisi anak terindikasi aman, tetap pantau perkembangannya.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob[1]*100,
            title={'text': "Risiko Putus Sekolah (%)", 'font': {'size': 16, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                'bar': {'color': '#f87171' if prob[1] > 0.5 else '#4ade80'},
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(34,197,94,0.3)'},
                    {'range': [33, 66], 'color': 'rgba(245,158,11,0.3)'},
                    {'range': [66, 100], 'color': 'rgba(220,38,38,0.3)'}
                ],
                'threshold': {'line': {'color': '#ffd700', 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=50, b=20),
                                 font=dict(color='white'))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Tampilkan detail faktor risiko
        if pred == 1:
            st.markdown("""
            <div class='info-box'>
            <b>⚠️ Faktor Risiko yang Teridentifikasi:</b><br>
            • Pendapatan keluarga sangat rendah (&lt; Rp 500.000)<br>
            • Jarak sekolah sangat jauh (&gt; 15 km)<br>
            • Minat belajar sangat rendah<br>
            • Dukungan orang tua sangat kurang<br>
            • Jam kerja anak tinggi (&gt; 20 jam/minggu)<br>
            • Kondisi lingkungan terpencil<br>
            • Fasilitas belajar tidak memadai<br>
            • Transportasi tidak tersedia
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-header'>📋 Dataset Lengkap</div>", unsafe_allow_html=True)
    st.dataframe(df.drop(columns=['Label']).head(20), use_container_width=True)
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv_data, "dataset_putus_sekolah_2026.csv", "text/csv")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="group-name">👩‍🎓 Kelompok Penelitian Data Science</div>
    <div style="margin:0.4rem 0">
        Gladis Primadona (2024020179) &nbsp;·&nbsp; Aulia Virgara (2024020230) &nbsp;·&nbsp; Jesika Tarigan (2024020119)
    </div>
    <div>Analisis Penentu Tingkat Putus Sekolah di Daerah Pedesaan · Random Forest · 2026</div>
</div>
""", unsafe_allow_html=True)
