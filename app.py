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
    background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(220,38,38,0.05));
    border: 1px solid rgba(220,38,38,0.3);
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
    le = LabelEncoder()
    df_enc = df.copy()
    cat_cols = df_enc.select_dtypes(include='object').columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['ID','Nama_Responden','Status_Putus_Sekolah']]
    encoders = {}
    for col in cat_cols:
        le_col = LabelEncoder()
        df_enc[col] = le_col.fit_transform(df_enc[col])
        encoders[col] = le_col

    feature_cols = [c for c in df_enc.columns if c not in
                    ['ID','Nama_Responden','Desa','Kecamatan','Status_Putus_Sekolah','Label']]
    X = df_enc[feature_cols]
    y = df_enc['Label']
    return X, y, feature_cols, encoders

df = load_data()
X, y, feature_cols, encoders = prepare_model_data(df)

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

    model_choice = "Random Forest"
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
# TAB 1 – EKSPLORASI DATA
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
            hovertemplate="<b>%{label}</b><br>%{value} responden<br>%{percent}<extra></extra>"
        ))
        fig_pie.add_annotation(text=f"<b>{pct_putus:.0f}%</b><br>Putus", x=0.5, y=0.5,
                                font=dict(size=14, color='white'),
                                showarrow=False)
        fig_pie.update_layout(title="Distribusi Status Putus Sekolah",
                               title_font=dict(size=15, color='white'),
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               height=340, font=dict(color='white'))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("📌 **Penjelasan:** Dari 100 responden, terdapat {:.0f}% anak yang putus sekolah.".format(pct_putus))

    with col2:
        fig_age = px.histogram(df, x='Usia_Anak', color='Status_Putus_Sekolah',
                                color_discrete_map={'Ya': RED, 'Tidak': GREEN},
                                nbins=11, barmode='overlay', opacity=0.8)
        fig_age.update_layout(title="Distribusi Usia Responden",
                               title_font=dict(size=15, color='white'),
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               height=340, font=dict(color='white'))
        st.plotly_chart(fig_age, use_container_width=True)
        st.caption("📌 **Penjelasan:** Anak usia 12-15 tahun memiliki risiko putus sekolah lebih tinggi.")

    col3, col4 = st.columns(2)
    with col3:
        gen = df.groupby(['Jenis_Kelamin', 'Status_Putus_Sekolah']).size().reset_index(name='count')
        fig_gen = px.bar(gen, x='Jenis_Kelamin', y='count', color='Status_Putus_Sekolah',
                          color_discrete_map={'Ya': RED, 'Tidak': GREEN}, barmode='group',
                          text='count')
        fig_gen.update_traces(textposition='outside')
        fig_gen.update_layout(title="Putus Sekolah per Jenis Kelamin",
                               title_font=dict(size=15, color='white'),
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               height=350, font=dict(color='white'))
        st.plotly_chart(fig_gen, use_container_width=True)
        st.caption("📌 **Penjelasan:** Anak laki-laki cenderung lebih banyak putus sekolah.")

    with col4:
        kelas_count = df[df['Status_Putus_Sekolah']=='Ya']['Kelas_Terakhir'].value_counts().head(6)
        kelas_df = pd.DataFrame({'Kelas': kelas_count.index, 'Jumlah': kelas_count.values})
        fig_kelas = px.bar(kelas_df, x='Kelas', y='Jumlah', color='Jumlah',
                            color_continuous_scale='reds', text='Jumlah')
        fig_kelas.update_traces(textposition='outside')
        fig_kelas.update_layout(title="Kelas Saat Putus Sekolah",
                                 title_font=dict(size=15, color='white'),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                 height=350, xaxis_tickangle=-45, coloraxis_showscale=False,
                                 font=dict(color='white'))
        st.plotly_chart(fig_kelas, use_container_width=True)
        st.caption("📌 **Penjelasan:** Kebanyakan putus sekolah terjadi pada jenjang SMP (Kelas 7-9).")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – VISUALISASI FAKTOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-header'>📈 Analisis Per Faktor Penentu</div>", unsafe_allow_html=True)

    faktor_tab = st.radio("Pilih Faktor:", ["💰 Ekonomi", "🌿 Lingkungan", "👨‍👩‍👧 Pekerjaan Orang Tua",
                                              "🛣️ Jalan & Jarak", "💡 Minat & Motivasi"],
                           horizontal=True)

    def bar_chart(col, title, order=None):
        grp = df.groupby([col, 'Status_Putus_Sekolah']).size().reset_index(name='count')
        if order:
            grp[col] = pd.Categorical(grp[col], categories=order, ordered=True)
            grp = grp.sort_values(col)
        fig = px.bar(grp, x=col, y='count', color='Status_Putus_Sekolah',
                     color_discrete_map={'Ya': RED, 'Tidak': GREEN}, barmode='group',
                     text='count')
        fig.update_traces(textposition='outside')
        fig.update_layout(title=title, title_font=dict(size=14, color='white'),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          height=350, xaxis_tickangle=-20, font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

    if faktor_tab == "💰 Ekonomi":
        c1, c2 = st.columns(2)
        with c1:
            fig_inc = go.Figure()
            for s, color in [('Ya', RED), ('Tidak', GREEN)]:
                d = df[df['Status_Putus_Sekolah']==s]['Pendapatan_Keluarga_Bulan']
                fig_inc.add_trace(go.Box(y=d, name=s, marker_color=color, boxmean=True))
            fig_inc.update_layout(title="Pendapatan vs Status Putus",
                                   title_font=dict(size=14, color='white'),
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   height=350, font=dict(color='white'))
            st.plotly_chart(fig_inc, use_container_width=True)
            st.caption("📌 **Penjelasan:** Pendapatan rendah meningkatkan risiko putus sekolah.")
        with c2:
            bar_chart('Kepemilikan_Lahan', 'Kepemilikan Lahan vs Putus Sekolah')
            st.caption("📌 **Penjelasan:** Keluarga tanpa lahan lebih rentan putus sekolah.")
        c3, c4 = st.columns(2)
        with c3:
            bar_chart('Status_Bansos', 'Penerima Bansos vs Putus Sekolah')
            st.caption("📌 **Penjelasan:** Penerima bansos masih perlu intervensi tambahan.")
        with c4:
            fig_tang = go.Figure()
            for s, color in [('Ya', RED), ('Tidak', GREEN)]:
                d = df[df['Status_Putus_Sekolah']==s]['Jumlah_Tanggungan_Keluarga']
                fig_tang.add_trace(go.Histogram(x=d, name=s, marker_color=color, opacity=0.75))
            fig_tang.update_layout(title="Jumlah Tanggungan vs Putus Sekolah", barmode='overlay',
                                    title_font=dict(size=14, color='white'),
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    height=350, font=dict(color='white'))
            st.plotly_chart(fig_tang, use_container_width=True)
            st.caption("📌 **Penjelasan:** Tanggungan >4 orang meningkatkan risiko.")

    elif faktor_tab == "🌿 Lingkungan":
        c1, c2 = st.columns(2)
        ord_kondisi = ["Sangat Terpencil","Terpencil","Cukup Terjangkau","Terjangkau"]
        with c1:
            bar_chart('Kondisi_Lingkungan', 'Kondisi Lingkungan vs Putus Sekolah', ord_kondisi)
            st.caption("📌 **Penjelasan:** Lingkungan terpencil berkorelasi dengan putus sekolah.")
        with c2:
            bar_chart('Akses_Listrik', 'Akses Listrik vs Putus Sekolah')
            st.caption("📌 **Penjelasan:** Akses listrik buruk menghambat belajar.")
        c3, c4 = st.columns(2)
        with c3:
            bar_chart('Akses_Internet', 'Akses Internet vs Putus Sekolah')
            st.caption("📌 **Penjelasan:** Keterbatasan akses internet memperlebar kesenjangan.")
        with c4:
            ord_fas = ["Sangat Kurang","Kurang","Cukup","Baik"]
            bar_chart('Ketersediaan_Fasilitas_Belajar', 'Fasilitas Belajar vs Putus Sekolah', ord_fas)
            st.caption("📌 **Penjelasan:** Fasilitas memadai menjadi faktor protektif.")

    elif faktor_tab == "👨‍👩‍👧 Pekerjaan Orang Tua":
        c1, c2 = st.columns(2)
        with c1:
            bar_chart('Pekerjaan_Ayah', 'Pekerjaan Ayah vs Putus Sekolah')
            st.caption("📌 **Penjelasan:** Pekerjaan ayah tidak tetap meningkatkan risiko.")
        with c2:
            bar_chart('Pekerjaan_Ibu', 'Pekerjaan Ibu vs Putus Sekolah')
            st.caption("📌 **Penjelasan:** Edukasi ibu tentang pendidikan sangat penting.")
        c3, c4 = st.columns(2)
        ord_pend = ["Tidak Sekolah","SD","SMP","SMA","Diploma/S1"]
        with c3:
            bar_chart('Pendidikan_Ayah', 'Pendidikan Ayah vs Putus Sekolah', ord_pend)
            st.caption("📌 **Penjelasan:** Pendidikan ayah rendah berkorelasi dengan putus sekolah.")
        with c4:
            fig_jam = go.Figure()
            for s, color in [('Ya', RED), ('Tidak', GREEN)]:
                d = df[df['Status_Putus_Sekolah']==s]['Jam_Kerja_Anak_Per_Minggu']
                fig_jam.add_trace(go.Box(y=d, name=s, marker_color=color, boxmean=True))
            fig_jam.update_layout(title="Jam Kerja Anak per Minggu",
                                   title_font=dict(size=14, color='white'),
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   height=350, font=dict(color='white'))
            st.plotly_chart(fig_jam, use_container_width=True)
            st.caption("📌 **Penjelasan:** Anak bekerja >15 jam/minggu berisiko tinggi.")

    elif faktor_tab == "🛣️ Jalan & Jarak":
        c1, c2 = st.columns(2)
        with c1:
            fig_jarak = px.scatter(df, x='Jarak_ke_Sekolah_km', y='Waktu_Tempuh_Menit',
                                    color='Status_Putus_Sekolah',
                                    color_discrete_map={'Ya': RED, 'Tidak': GREEN},
                                    size='Jumlah_Tanggungan_Keluarga', opacity=0.8)
            fig_jarak.update_layout(title="Jarak & Waktu Tempuh vs Putus Sekolah",
                                     title_font=dict(size=14, color='white'),
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     height=350, font=dict(color='white'))
            st.plotly_chart(fig_jarak, use_container_width=True)
            st.caption("📌 **Penjelasan:** Jarak >5 km dan waktu >60 menit meningkatkan risiko.")
        with c2:
            ord_jalan = ["Jalan Tanah","Jalan Kerikil","Jalan Aspal Rusak","Jalan Aspal Baik"]
            bar_chart('Jenis_Jalan', 'Jenis Jalan vs Putus Sekolah', ord_jalan)
            st.caption("📌 **Penjelasan:** Jalan rusak menyulitkan akses ke sekolah.")
        c3, c4 = st.columns(2)
        with c3:
            bar_chart('Ketersediaan_Transportasi', 'Transportasi vs Putus Sekolah')
            st.caption("📌 **Penjelasan:** Tidak ada transportasi menjadi hambatan utama.")
        with c4:
            ord_hujan = ["Tidak Bisa Dilalui","Sangat Sulit","Sulit","Bisa Dilalui"]
            bar_chart('Kondisi_Jalan_Saat_Hujan', 'Kondisi Jalan Hujan vs Putus Sekolah', ord_hujan)
            st.caption("📌 **Penjelasan:** Jalan tidak bisa dilalui saat hujan menyebabkan absensi.")

    else:
        c1, c2 = st.columns(2)
        ord_minat = ["Sangat Rendah","Rendah","Sedang","Tinggi","Sangat Tinggi"]
        with c1:
            bar_chart('Minat_Belajar_Anak', 'Minat Belajar vs Putus Sekolah', ord_minat)
            st.caption("📌 **Penjelasan:** Minat rendah adalah prediktor kuat putus sekolah.")
        with c2:
            ord_dukung = ["Sangat Kurang","Kurang","Cukup","Baik","Sangat Baik"]
            bar_chart('Dukungan_Orang_Tua', 'Dukungan Orang Tua vs Putus Sekolah', ord_dukung)
            st.caption("📌 **Penjelasan:** Dukungan orang tua adalah faktor protektif terkuat.")
        c3, c4 = st.columns(2)
        with c3:
            bar_chart('Motivasi_Melanjutkan_Sekolah', 'Motivasi Sekolah vs Putus Sekolah', ord_minat)
            st.caption("📌 **Penjelasan:** Motivasi rendah membuat anak mudah menyerah.")
        with c4:
            ord_teman = ["Sangat Negatif","Negatif","Netral","Positif","Sangat Positif"]
            bar_chart('Pengaruh_Teman_Sebaya', 'Pengaruh Teman vs Putus Sekolah', ord_teman)
            st.caption("📌 **Penjelasan:** Lingkungan teman negatif mendorong putus sekolah.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – HASIL MODEL (DIPERBAIKI - TANPA STYLER)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-header'>🤖 Hasil Model: Random Forest</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Evaluasi performa model data mining untuk klasifikasi putus sekolah</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <b>🌲 Mengapa Random Forest?</b><br>
    • <b>Akurasi tinggi</b> - ensemble method menggabungkan banyak pohon keputusan<br>
    • <b>Tahan overfitting</b> - random sampling mengurangi overfitting<br>
    • <b>Feature importance</b> - dapat mengidentifikasi faktor paling berpengaruh
    </div>
    """, unsafe_allow_html=True)

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
        fig_cm.update_layout(title="Confusion Matrix",
                              title_font=dict(size=16, color='white'),
                              paper_bgcolor='rgba(0,0,0,0)', height=400,
                              font=dict(color='white'))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("📌 **Penjelasan:** Nilai diagonal menunjukkan prediksi benar (TN dan TP).")

    with col2:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Random Forest (AUC = {roc_auc:.3f})',
                                      line=dict(color='#ffd700', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Classifier',
                                      line=dict(color='#94a3b8', width=2, dash='dash')))
        fig_roc.update_layout(title=f"ROC Curve (AUC = {roc_auc:.3f})",
                               title_font=dict(size=16, color='white'),
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400,
                               xaxis=dict(title='False Positive Rate', gridcolor='#334155', color='white'),
                               yaxis=dict(title='True Positive Rate', gridcolor='#334155', color='white'),
                               font=dict(color='white'))
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption("📌 **Penjelasan:** AUC = 1.000 menunjukkan klasifikasi sempurna.")

    st.markdown("<div class='section-header'>🔍 Feature Importance (Top 10)</div>", unsafe_allow_html=True)
    fi_top = fi.head(10).copy()
    fi_top['feature_clean'] = fi_top['feature'].str.replace('_', ' ')

    fig_fi = go.Figure(go.Bar(
        x=fi_top['importance'], y=fi_top['feature_clean'],
        orientation='h',
        marker=dict(color=fi_top['importance'], colorscale=[[0,'#3b82f6'],[1,'#ffd700']], showscale=False),
        text=[f"{v:.3f}" for v in fi_top['importance']],
        textposition='outside',
        textfont=dict(color='white')
    ))
    fig_fi.update_layout(title="10 Fitur Paling Berpengaruh terhadap Putus Sekolah",
                          title_font=dict(size=16, color='white'),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450,
                          margin=dict(l=200), xaxis=dict(title='Importance Score', gridcolor='#334155', color='white'),
                          yaxis=dict(color='white', gridcolor='#334155'),
                          font=dict(color='white'))
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption("📌 **Penjelasan:** Fitur dengan importance tertinggi adalah faktor paling dominan.")

    st.markdown("<div class='section-header'>📊 Laporan Klasifikasi</div>", unsafe_allow_html=True)
    report = classification_report(y_te, y_pred, target_names=['Tidak Putus', 'Putus Sekolah'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(3)
    
    # TAMPILKAN TANPA STYLER (HANYA DATAFRAME BIASA)
    st.dataframe(report_df, use_container_width=True)
    
    # Tambahkan penjelasan metrik
    st.markdown("""
    <div class='info-box'>
    <b>📖 Penjelasan Metrik Evaluasi:</b><br>
    • <b>Precision:</b> Dari yang diprediksi putus, berapa yang benar putus = TP/(TP+FP)<br>
    • <b>Recall:</b> Dari yang benar putus, berapa yang terdeteksi = TP/(TP+FN)<br>
    • <b>F1-Score:</b> Rata-rata harmonik precision dan recall = 2 × (P×R)/(P+R)<br>
    • <b>Accuracy:</b> Total prediksi benar = (TP+TN)/(Total)<br>
    • <b>Support:</b> Jumlah sampel aktual untuk setiap kelas
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-header'>🔮 Prediksi Individual</div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("**📋 Data Identitas**")
        c1, c2, c3 = st.columns(3)
        with c1: usia = st.slider("Usia Anak", 7, 18, 12)
        with c2: jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        with c3: kelas = st.selectbox("Kelas Terakhir", ["SD Kelas 1","SD Kelas 2","SD Kelas 3","SD Kelas 4","SD Kelas 5","SD Kelas 6","SMP Kelas 7","SMP Kelas 8","SMP Kelas 9","SMA Kelas 10","SMA Kelas 11","SMA Kelas 12"])

        st.markdown("**💰 Faktor Ekonomi**")
        c4, c5, c6, c7 = st.columns(4)
        with c4: pendapatan = st.number_input("Pendapatan/Bulan (Rp)", 300000, 5000000, 800000, 50000)
        with c5: tanggungan = st.slider("Jumlah Tanggungan", 1, 10, 4)
        with c6: lahan = st.selectbox("Kepemilikan Lahan", ["Tidak Punya","Sewa","Milik Sendiri"])
        with c7: bansos = st.selectbox("Penerima Bansos", ["Ya","Tidak"])

        st.markdown("**🌿 Faktor Lingkungan**")
        c8, c9, c10, c11 = st.columns(4)
        with c8: kondisi_ling = st.selectbox("Kondisi Lingkungan", ["Sangat Terpencil","Terpencil","Cukup Terjangkau","Terjangkau"])
        with c9: listrik = st.selectbox("Akses Listrik", ["Ya","Tidak"])
        with c10: internet = st.selectbox("Akses Internet", ["Ya","Tidak"])
        with c11: fasilitas = st.selectbox("Fasilitas Belajar", ["Sangat Kurang","Kurang","Cukup","Baik"])

        st.markdown("**👨‍👩‍👧 Pekerjaan Orang Tua**")
        c12, c13, c14, c15, c16 = st.columns(5)
        with c12: pekerjaan_ayah = st.selectbox("Pekerjaan Ayah", ["Petani","Buruh Harian","Nelayan","Pedagang Kecil","Pekerja Serabutan","Tidak Bekerja"])
        with c13: pekerjaan_ibu = st.selectbox("Pekerjaan Ibu", ["Ibu Rumah Tangga","Petani","Buruh","Pedagang Kecil","Tidak Bekerja"])
        with c14: pend_ayah = st.selectbox("Pend. Ayah", ["Tidak Sekolah","SD","SMP","SMA","Diploma/S1"])
        with c15: pend_ibu = st.selectbox("Pend. Ibu", ["Tidak Sekolah","SD","SMP","SMA","Diploma/S1"])
        with c16: jam_kerja = st.slider("Jam Kerja Anak/Minggu", 0, 50, 5)

        st.markdown("**🛣️ Jalan & Jarak**")
        c17, c18, c19, c20, c21 = st.columns(5)
        with c17: jarak = st.slider("Jarak Sekolah (km)", 1.0, 60.0, 10.0, 0.5)
        with c18: jenis_jalan = st.selectbox("Jenis Jalan", ["Jalan Tanah","Jalan Kerikil","Jalan Aspal Rusak","Jalan Aspal Baik"])
        with c19: waktu = st.slider("Waktu Tempuh (mnt)", 5, 180, 30)
        with c20: transportasi = st.selectbox("Transportasi", ["Tidak Ada","Ojek/Becak","Angkutan Umum","Kendaraan Pribadi"])
        with c21: cond_hujan = st.selectbox("Jalan Saat Hujan", ["Tidak Bisa Dilalui","Sangat Sulit","Sulit","Bisa Dilalui"])

        st.markdown("**💡 Minat & Motivasi**")
        c22, c23, c24, c25 = st.columns(4)
        with c22: minat = st.selectbox("Minat Belajar", ["Sangat Rendah","Rendah","Sedang","Tinggi","Sangat Tinggi"])
        with c23: dukungan = st.selectbox("Dukungan Orang Tua", ["Sangat Kurang","Kurang","Cukup","Baik","Sangat Baik"])
        with c24: motivasi = st.selectbox("Motivasi Sekolah", ["Sangat Rendah","Rendah","Sedang","Tinggi","Sangat Tinggi"])
        with c25: teman = st.selectbox("Pengaruh Teman", ["Sangat Negatif","Negatif","Netral","Positif","Sangat Positif"])

        submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

    if submitted:
        input_dict = {
            'Usia_Anak': usia, 'Jenis_Kelamin': jk, 'Kelas_Terakhir': kelas,
            'Pendapatan_Keluarga_Bulan': pendapatan, 'Jumlah_Tanggungan_Keluarga': tanggungan,
            'Kepemilikan_Lahan': lahan, 'Status_Bansos': bansos,
            'Kondisi_Lingkungan': kondisi_ling, 'Akses_Listrik': listrik,
            'Akses_Internet': internet, 'Ketersediaan_Fasilitas_Belajar': fasilitas,
            'Pekerjaan_Ayah': pekerjaan_ayah, 'Pekerjaan_Ibu': pekerjaan_ibu,
            'Pendidikan_Ayah': pend_ayah, 'Pendidikan_Ibu': pend_ibu,
            'Jam_Kerja_Anak_Per_Minggu': jam_kerja,
            'Jarak_ke_Sekolah_km': jarak, 'Jenis_Jalan': jenis_jalan,
            'Waktu_Tempuh_Menit': waktu, 'Ketersediaan_Transportasi': transportasi,
            'Kondisi_Jalan_Saat_Hujan': cond_hujan,
            'Minat_Belajar_Anak': minat, 'Dukungan_Orang_Tua': dukungan,
            'Motivasi_Melanjutkan_Sekolah': motivasi, 'Pengaruh_Teman_Sebaya': teman
        }
        input_df = pd.DataFrame([input_dict])
        for col in input_df.select_dtypes(include='object').columns:
            if col in encoders:
                try:
                    input_df[col] = encoders[col].transform(input_df[col])
                except:
                    input_df[col] = 0
        input_df = input_df[feature_cols]
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        if pred == 1:
            st.markdown(f"""
            <div class='pred-result pred-danger'>
                <div class='pred-title'>⚠️ BERISIKO PUTUS SEKOLAH</div>
                <div style='font-size:1rem; margin-top:0.5rem; color:#cbd5e1'>
                    Probabilitas Putus Sekolah: <b style='color:#f87171'>{prob[1]*100:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='pred-result pred-safe'>
                <div class='pred-title'>✅ AMAN — TIDAK BERISIKO</div>
                <div style='font-size:1rem; margin-top:0.5rem; color:#cbd5e1'>
                    Probabilitas Melanjutkan Sekolah: <b style='color:#4ade80'>{prob[0]*100:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob[1]*100,
            title={'text': "Risiko Putus Sekolah (%)", 'font': {'size': 16, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                'bar': {'color': RED if prob[1] > 0.5 else GREEN},
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(34,197,94,0.2)'},
                    {'range': [33, 66], 'color': 'rgba(245,158,11,0.2)'},
                    {'range': [66, 100], 'color': 'rgba(220,38,38,0.2)'}
                ],
                'threshold': {'line': {'color': '#ffd700', 'width': 3}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280,
                                 font=dict(color='white'))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-header'>📋 Dataset Lengkap</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        filter_status = st.selectbox("Filter Status", ["Semua", "Putus Sekolah", "Tidak Putus"])
    with c2:
        filter_gender = st.selectbox("Filter Jenis Kelamin", ["Semua", "Laki-laki", "Perempuan"])
    with c3:
        search = st.text_input("🔍 Cari Desa/Kecamatan")

    df_show = df.copy()
    if filter_status != "Semua":
        status_val = "Ya" if filter_status == "Putus Sekolah" else "Tidak"
        df_show = df_show[df_show['Status_Putus_Sekolah'] == status_val]
    if filter_gender != "Semua":
        df_show = df_show[df_show['Jenis_Kelamin'] == filter_gender]
    if search:
        df_show = df_show[df_show['Desa'].str.contains(search, case=False) |
                          df_show['Kecamatan'].str.contains(search, case=False)]

    st.markdown(f"<div class='info-box'>Menampilkan <b>{len(df_show)}</b> dari <b>{len(df)}</b> data</div>", unsafe_allow_html=True)
    
    st.dataframe(df_show.drop(columns=['Label']).reset_index(drop=True), use_container_width=True, height=500)

    csv_data = df_show.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv_data, "dataset_putus_sekolah_2026.csv", "text/csv", use_container_width=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="group-name">👩‍🎓 Kelompok Penelitian Data Science</div>
    <div style="margin:0.4rem 0">
        Gladis Primadona (2024020179) &nbsp;·&nbsp; Aulia Virgara (2024020230) &nbsp;·&nbsp; Jesika Tarigan (2024020119)
    </div>
    <div style="margin-top:0.4rem">
        Analisis Penentu Tingkat Putus Sekolah di Daerah Pedesaan · Metode: Random Forest · 2026
    </div>
</div>
""", unsafe_allow_html=True)
