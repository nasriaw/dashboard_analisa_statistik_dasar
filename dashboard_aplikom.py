import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import pingouin as pg

# --- 1. KONFIGURASI HALAMAN & CSS ---
st.set_page_config(page_title="Aplikasi Statistika Dasar - STIEI Malang", layout="wide")

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px; border-radius: 10px; color: white; text-align: center; margin-bottom: 5px;
    }
    .author-sub {
        text-align: center; font-style: italic; color: #555; margin-bottom: 30px; font-size: 1.1em;
    }
    .result-box {
        background-color: #f0f2f6; padding: 20px; border-radius: 10px;
        border-left: 5px solid #007bff; margin: 10px 0;
    }
    .instruction-box {
        background-color: #fff4e5; padding: 15px; border-radius: 8px;
        border: 1px solid #ffa94d; margin-bottom: 20px; font-size: 0.95em;
    }
    .footer-text { font-size: 12px; color: gray; text-align: center; margin-top: 50px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MANAJEMEN DATA (SESSION STATE) ---
if 'df' not in st.session_state:
    try:
        # Load data default awal
        st.session_state.df = pd.read_csv('data_50_kepuasaan_client.csv')
    except:
        # Fallback dummy data jika file tidak ditemukan
        data = np.random.randint(10, 100, size=(50, 6))
        st.session_state.df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'X4', 'X5', 'Y'])

def render_header(title):
    st.markdown(f'<div class="main-header"><h1>{title}</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="author-sub">Penyusun: Ir. M Nasri AW, M.Eng.Sc, M.Kom - Dosen STIE Indonesia Malang @2025</div>', unsafe_allow_html=True)

def handle_upload(key):
    st.subheader("b. Contoh Data (Head)")
    st.write("Data yang sedang digunakan (Preview 10 baris):")
    st.dataframe(st.session_state.df.head(10))
    
    st.markdown("""
    <div class="instruction-box">
    ⚠️ <b>Ketentuan Analisis Data:</b><br>
    - Gunakan file format <b>.CSV</b>.<br>
    - Variabel <b>Y (Dependen)</b>: Harus kolom paling <b>KANAN</b>.<br>
    - Variabel <b>X (Independen)</b>: Semua kolom dari pertama sampai sebelum terakhir.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📂 Browse File CSV / Upload Data Baru", type=["csv"], key=key)
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("Data berhasil diperbarui!")
        st.rerun()

# Deteksi Kolom Dinamis
df = st.session_state.df
all_cols = df.columns.tolist()
x_vars_all = all_cols[:-1]  # Kolom pertama sampai N-1
y_var_final = all_cols[-1]   # Kolom terakhir

# --- 3. SIDEBAR NAVIGASI ---
with st.sidebar:
    st.markdown("### 📋 Menu Solver")
    menu = st.radio("Pilih Analisis:", 
                    ["Home", "Analisis Deskripsi", "Visualisasi Graph", "Korelasi", "Regresi & Prediksi", "ANOVA (1 & 2 Way)", "Uji Hipotesis"])

# --- 4. LOGIKA ANALISIS ---

# --- HOME ---
if menu == "Home":
    render_header("🔧 Aplikasi Python Untuk Statistika Dasar")
    st.subheader("a. Review Materi Statistika Dasar")
    st.write("Aplikasi Python ini lanjutan Kuliah Economic Statistics bagi mahasiswa S1 Manajemen STIEI Malang, dari aspek penerapan aplikasi komputer menggunakan bahasa python." \
    " Bagian ini mengulang singkat materi utama pada kuliah EConomic Statistics yang meliputi:")
    st.markdown("### Konsep Data dan Sample")
    st.markdown("Data adalah informasi yang diorganisir dalam bentuk yang dapat dianalisis. Sample adalah subset dari populasi yang digunakan untuk membuat kesimpulan tentang populasi secara keseluruhan.")
    
    st.markdown("### Populasi, Sampling dan Teknik Sampling")
    st.markdown("Populasi adalah keseluruhan subjek yang diteliti. Sampling adalah proses pemilihan sample dari populasi. Teknik sampling meliputi:")
    st.markdown("- Random Sampling")
    st.markdown("- Stratified Sampling")
    st.markdown("- Cluster Sampling")
    st.markdown("- Systematic Sampling")
    
    st.markdown("### Distribusi")
    st.markdown("Distribusi statistik menggambarkan bagaimana data tersebar. Contoh: distribusi normal, distribusi t, distribusi chi-square.")
    
    st.markdown("### Tipe dan Kegunaan Analisis Statistik")
    st.markdown("- **Analisis Deskriptif:** Ringkasan data (mean, median, modus)")
    st.markdown("- **Analisis Inferensial:** Kesimpulan tentang populasi dari data sample")
    st.markdown("- **Analisis Prediktif:** Memprediksi nilai berdasarkan data")
    
    st.markdown("### Atribut Dasar Olah Data")
    st.markdown("- Tipe data: numerik, kategorikal")
    st.markdown("- Skala pengukuran: nominal, ordinal, interval, rasio")
    
    st.markdown("### Pembersihan Data (Data Cleaning)")
    st.markdown("Proses mengidentifikasi dan memperbaiki kesalahan dalam dataset untuk meningkatkan kualitas data.")
    
    st.markdown("### Visualisasi Data")
    st.markdown("- Scatter: Hubungan antar variabel")
    st.markdown("- Line: Tren data")
    st.markdown("- Bar: Perbandingan kategori")
    st.markdown("- Boxplot: Distribusi dan outlier")
    
    st.markdown("### Statistik Deskripsi dan Inferensi")
    st.markdown("- **Deskripsi:** Ringkasan statistik")
    st.markdown("- **Inferensi:** Pengujian hipotesis, estimasi parameter")
    
    st.markdown("### Korelasi")
    st.markdown("Ukuran kekuatan dan arah hubungan antara dua variabel.")
    
    st.markdown("### Regresi")
    st.markdown("Model untuk memprediksi nilai variabel dependen berdasarkan variabel independen.")
    
    st.markdown("### Uji Hipotesis")
    st.markdown("Prosedur untuk membuat keputusan tentang populasi berdasarkan data sample.")

    handle_upload("home_up")

# --- ANALISIS DESKRIPSI ---
elif menu == "Analisis Deskripsi":
    render_header("📑 Analisis Deskripsi")
    st.subheader("a. Pengertian Singkat")
    st.write("Memberikan ringkasan statistik seperti rata-rata, standar deviasi, skewness, dan kurtosis.")
    handle_upload("desc_up")
    
    st.subheader("d. Output")
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        desc = num_df.describe().T
        desc['skewness'] = num_df.skew()
        desc['kurtosis'] = num_df.kurtosis()
        st.dataframe(desc.style.format("{:.2f}"))
    else:
        st.error("Data tidak mengandung kolom numerik.")

# --- VISUALISASI ---
elif menu == "Visualisasi Graph":
    render_header("📊 Visualisasi Graph")
    st.subheader("a. Pengertian Singkat")
    st.write("Melihat pola dan sebaran data melalui representasi grafis.")
    handle_upload("viz_up")
    
    st.subheader("c. Template Isian Input")
    tipe_v = st.selectbox("Pilih Tipe Grafik:", ["scatter", "histplot", "boxplot"])
    col_sel = st.selectbox("Pilih Variabel X:", all_cols)
    
    st.subheader("d. Output")
    try:
        if tipe_v == "scatter":
            fig = px.scatter(df, x=col_sel, y=y_var_final, trendline="ols", title=f"Plot {col_sel} vs {y_var_final}")
        elif tipe_v == "histplot":
            fig = px.histogram(df, x=col_sel, color_discrete_sequence=['#1e3c72'])
        else:
            fig = px.box(df, y=col_sel)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("Gagal memproses grafik. Pastikan data numerik.")

# --- KORELASI ---
elif menu == "Korelasi":
    render_header("🔗 Analisis Korelasi")
    st.subheader("a. Pengertian Singkat")
    st.write("Mengukur kekuatan hubungan linear antar variabel.")
    handle_upload("corr_up")
    
    st.subheader("c. Template Isian Input")
    method = st.radio("Metode:", ["pearson", "spearman"])
    
    st.subheader("d. Output")
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        corr_matrix = num_df.corr(method=method)
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Tidak ada data numerik.")

# --- REGRESI & PREDIKSI (FIXED) ---
elif menu == "Regresi & Prediksi":
    render_header("📈 Analisis Regresi & Simulasi Prediksi")
    st.subheader("a. Pengertian Singkat")
    st.write(f"Menganalisis pengaruh variabel X terhadap {y_var_final} (Kolom Paling Kanan).")
    handle_upload("reg_up")
    
    st.subheader("c. Template Isian Input")
    mode = st.radio("Pilih Mode Regresi:", ["Sederhana", "Berganda"])
    
    # Logika Pemilihan Variabel X
    if mode == "Sederhana":
        selected_x = st.selectbox("Pilih 1 Variabel Independen (X):", x_vars_all)
        X_final = df[[selected_x]]
    else:
        X_final = df[x_vars_all]
        st.info(f"Menggunakan semua variabel X: {', '.join(x_vars_all)}")

    st.subheader("d. Output Analisis")
    try:
        # Kalkulasi Model
        X_const = sm.add_constant(X_final)
        model = sm.OLS(df[y_var_final], X_const).fit()
        
        # 1. Tampilkan Persamaan Matematis
        st.markdown("### 1. Persamaan Regresi")
        p = model.params
        persamaan = rf"Y = {p.iloc[0]:.4f}"
        for i, col in enumerate(X_final.columns):
            persamaan += rf" + ({p.iloc[i+1]:.4f} \cdot {col})"
        st.latex(persamaan)
        
        # 2. Ringkasan Statistik
        st.write(model.summary())
        
        # 3. Simulasi Prediksi
        st.markdown("### 2. 🔮 Simulasi Prediksi Y")
        inputs = []
        cols = st.columns(len(X_final.columns))
        for i, col_name in enumerate(X_final.columns):
            with cols[i]:
                v = st.number_input(f"Nilai {col_name}:", value=float(df[col_name].mean()))
                inputs.append(v)
        
        y_pred = p.iloc[0] + sum(np.array(inputs) * p.iloc[1:])
        st.markdown(f'<div class="result-box">Prediksi {y_var_final}: <b>{y_pred:.4f}</b></div>', unsafe_allow_html=True)

        # 4. Uji Asumsi Klasik
        st.markdown("### 3. Uji Asumsi Klasik")
        residu = model.resid
        sw_p = stats.shapiro(residu).pvalue
        dw = durbin_watson(residu)
        st.write(f"- **Uji Normalitas (Shapiro-Wilk):** P-Value = {sw_p:.4f} " + ("(Normal ✅)" if sw_p > 0.05 else "(Tidak Normal ❌)"))
        st.write(f"- **Uji Autokorelasi (Durbin-Watson):** {dw:.2f}")

    except Exception as e:
        st.error(f"Kesalahan analisis: {e}")

# --- ANOVA ---
elif menu == "ANOVA (1 & 2 Way)":
    render_header("🧪 Analisis Varians (ANOVA)")
    st.subheader("a. Pengertian Singkat")
    st.write(f"Menguji perbedaan rata-rata {y_var_final} antar kelompok.")
    handle_upload("ano_up")
    
    st.subheader("c. Template Isian Input")
    f1 = st.selectbox("Faktor 1 (Grup):", x_vars_all)
    mode_a = st.selectbox("Jenis ANOVA:", ["One-Way", "Two-Way"])
    
    st.subheader("d. Output")
    try:
        if mode_a == "One-Way":
            res_a = pg.anova(dv=y_var_final, between=f1, data=df)
        else:
            f2 = st.selectbox("Faktor 2 (Grup):", [c for c in x_vars_all if c != f1])
            res_a = pg.anova(dv=y_var_final, between=[f1, f2], data=df)
        st.dataframe(res_a)
    except:
        st.warning("Variabel faktor harus kategori (Grup).")

# --- HIPOTESIS ---
elif menu == "Uji Hipotesis":
    render_header("❓ Uji Hipotesis Parsial & Serentak")
    st.subheader("a. Pengertian Singkat")
    st.write("Menguji signifikansi pengaruh X terhadap Y secara mandiri (t) dan bersama-sama (F).")
    handle_upload("hyp_up")
    
    st.subheader("c. Template Isian Input")
    alpha = st.slider("Signifikansi (Alpha):", 0.01, 0.10, 0.05)
    
    st.subheader("d. Output Analisis")
    try:
        X_h = sm.add_constant(df[x_vars_all])
        mod_h = sm.OLS(df[y_var_final], X_h).fit()
        
        st.markdown("### 1. Uji Parsial (Uji t)")
        t_df = pd.DataFrame({
            'Koefisien': mod_h.params,
            'P-Value': mod_h.pvalues,
            'Hasil': ['Signifikan' if p < alpha else 'Tidak Signifikan' for p in mod_h.pvalues]
        })
        st.table(t_df)
        
        st.markdown("### 2. Uji Serentak (Uji F)")
        st.write(f"**P-Value F:** {mod_h.f_pvalue:.4e}")
        st.success("Signifikan" if mod_h.f_pvalue < alpha else "Tidak Signifikan")
    except:
        st.error("Analisis memerlukan kolom numerik.")

st.markdown('<div class="footer-text">Aplikasi Statistika Dasar @2025 - STIEI Malang</div>', unsafe_allow_html=True)
