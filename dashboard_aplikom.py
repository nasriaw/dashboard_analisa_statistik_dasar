import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('data_test_300_kepuasaan_client.csv')
    return data

data = load_data()

# Set page config
import streamlit as st
st.set_page_config(
   page_title="Aplikom Statistika Menggunakan Python",
   page_icon="📊",
   layout="wide",
   initial_sidebar_state="expanded",
   menu_items=None)

# Main content
st.title("Aplikom Statistika Menggunakan Python")
st.markdown("---")
st.markdown("**Penyusun:** Ir. M Nasri AW, M.Eng.Sc, M.Kom, Dosen STIE Indonesia Malang, @2025**")
st.markdown("**Python Library:** streamlit, pandas, numpy, statsmodels, scipy, scikit-learn, openpyxl, pingouin, plotly.express, seaborn")
st.markdown("---")

# Display data preview
st.subheader("Preview Data Asli")
st.dataframe(data) #.head())

# Menu selection using selectbox
st.markdown("### Pilih Menu Analisis")
menu_options = [
    "Pengertian Statistika",
    "Sampling",
    "Statistik Deskriptif",
    "Visualisasi Data",
    "Korelasi",
    "Regresi",
    "ANOVA",
    "Uji Hipotesis"
]

selected_menu = st.sidebar.radio("📋 Menu Navigasi:", menu_options)

# === MENU: PENGERTIAN STATISTIKA ===
if selected_menu == "Pengertian Statistika":
    st.header("📚 Pengertian Konsep Statistika")
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

# === MENU: SAMPLING ===
elif selected_menu == "Sampling":
    st.header("🔍 Teknik Sampling")
    st.markdown("Contoh implementasi sampling dalam dataset:")

    # Random sampling
    st.subheader("Random Sampling")
    sample_size = st.slider("Ukuran Sample", 10, 100, 30)
    random_sample = data.sample(sample_size)
    st.dataframe(random_sample)

    # Source code random sampling
    st.markdown("### Source Code Random Sampling")
    st.code("""
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('data_test_300_kepuasaan_client.csv')

# Random sampling
sample_size = 30
random_sample = data.sample(sample_size)
print(random_sample.head())
    """)

    # Stratified sampling
    st.subheader("Stratified Sampling (berdasarkan KepuasanKlien)")
    bins = [0, 40, 60, 80, 100]
    labels = ['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
    data['KategoriKepuasan'] = pd.cut(data['KepuasanKlien'], bins=bins, labels=labels)
    
    stratified_sample = data.groupby('KategoriKepuasan').apply(lambda x: x.sample(n=5, random_state=1)).reset_index(drop=True)
    st.dataframe(stratified_sample)

    # Source code stratified sampling
    st.markdown("### Source Code Stratified Sampling")
    st.code("""
import pandas as pd

# Load data
data = pd.read_csv('data_test_300_kepuasaan_client.csv')

# Create categories
bins = [0, 40, 60, 80, 100]
labels = ['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
data['KategoriKepuasan'] = pd.cut(data['KepuasanKlien'], bins=bins, labels=labels)

# Stratified sampling
stratified_sample = data.groupby('KategoriKepuasan').apply(lambda x: x.sample(n=5, random_state=1)).reset_index(drop=True)
print(stratified_sample.head())
    """)

# === MENU: STATISTIK DESKRIPTIF ===
elif selected_menu == "Statistik Deskriptif":
    st.header("📊 Statistik Deskriptif")
    st.markdown("Analisis statistik deskriptif untuk dataset:")
    
    # Select columns for analysis
    numeric_cols = ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja', 'KepuasanKlien']
    selected_col = st.selectbox("Pilih Variabel:", numeric_cols)
    
    # Calculate descriptive statistics
    desc_stats = data[selected_col].describe()
    st.subheader(f"Statistik Deskriptif untuk {selected_col}")
    st.dataframe(desc_stats)
    
    # Additional statistics
    st.markdown(f"**Median:** {data[selected_col].median()}")
    st.markdown(f"**Modus:** {data[selected_col].mode()[0]}")
    st.markdown(f"**Skewness:** {stats.skew(data[selected_col])}")
    st.markdown(f"**Kurtosis:** {stats.kurtosis(data[selected_col])}")
    
    # Histogram
    st.subheader("Distribusi Data")
    fig, ax = plt.subplots()
    sns.histplot(data[selected_col], kde=True, ax=ax)
    st.pyplot(fig)

    # Source code statistik deskriptif
    st.markdown("### Source Code Statistik Deskriptif")
    st.code("""
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data_test_300_kepuasaan_client.csv')

# Select variable
selected_col = 'KepuasanKlien'

# Calculate descriptive statistics
desc_stats = data[selected_col].describe()
print(desc_stats)

# Additional statistics
print(f"Median: {data[selected_col].median()}")
print(f"Modus: {data[selected_col].mode()[0]}")
print(f"Skewness: {stats.skew(data[selected_col])}")
print(f"Kurtosis: {stats.kurtosis(data[selected_col])}")

# Plot histogram
plt.figure()
sns.histplot(data[selected_col], kde=True)
plt.title(f"Distribusi {selected_col}")
plt.show()
    """)

# === MENU: VISUALISASI DATA ===
elif selected_menu == "Visualisasi Data":
    st.header("📈 Visualisasi Data")
    
    # Chart type selection
    chart_type = st.selectbox("Pilih Tipe Grafik:", 
                             ["Histogram", "Boxplot", "Scatter Plot", "Line Plot", "Bar Plot", "Heatmap"])
    
    if chart_type == "Histogram":
        numeric_cols = ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja', 'KepuasanKlien']
        selected_col = st.selectbox("Pilih Variabel:", numeric_cols)
        st.subheader(f"Histogram {selected_col}")
        fig = px.histogram(data, x=selected_col, nbins=20)
        st.plotly_chart(fig)
    
    elif chart_type == "Boxplot":
        numeric_cols = ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja', 'KepuasanKlien']
        selected_col = st.selectbox("Pilih Variabel:", numeric_cols)
        st.subheader(f"Boxplot {selected_col}")
        fig = px.box(data, y=selected_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("Pilih Variabel X:", ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja'])
        y_col = st.selectbox("Pilih Variabel Y:", ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja', 'KepuasanKlien'])
        st.subheader(f"Scatter Plot: {x_col} vs {y_col}")
        fig = px.scatter(data, x=x_col, y=y_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Line Plot":
        x_col = st.selectbox("Pilih Variabel X:", ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja'])
        y_col = st.selectbox("Pilih Variabel Y:", ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja', 'KepuasanKlien'])
        st.subheader(f"Line Plot: {x_col} vs {y_col}")
        fig = px.line(data, x=x_col, y=y_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Bar Plot":
        cat_col = st.selectbox("Pilih Variabel Kategori:", ['LamaPend'])
        num_col = st.selectbox("Pilih Variabel Numerik:", ['KepuasanKlien'])
        st.subheader(f"Bar Plot: {cat_col} vs {num_col}")
        fig = px.bar(data, x=cat_col, y=num_col, barmode='group')
        st.plotly_chart(fig)
    
    elif chart_type == "Heatmap":
        st.subheader("Heatmap Korelasi")
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Source code visualisasi
    st.markdown("### Source Code Visualisasi Data")
    st.code("""
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data_test_300_kepuasaan_client.csv')

# Histogram
fig_hist = px.histogram(data, x='KepuasanKlien', nbins=20)
fig_hist.show()

# Boxplot
fig_box = px.box(data, y='KepuasanKlien')
fig_box.show()

# Scatter plot
fig_scatter = px.scatter(data, x='JamKerja', y='KepuasanKlien')
fig_scatter.show()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
    """)

# === MENU: KORELASI ===
elif selected_menu == "Korelasi":
    st.header("🔗 Analisis Korelasi")
    
    # Correlation matrix
    st.subheader("Matriks Korelasi")
    corr_matrix = data.corr()
    st.dataframe(corr_matrix)
    
    # Visualisasi korelasi
    st.subheader("Visualisasi Korelasi")
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
    st.plotly_chart(fig)
    
    # Korelasi dengan variabel target
    target_var = st.selectbox("Pilih Variabel Target:", ['KepuasanKlien'])
    st.subheader(f"Korelasi dengan {target_var}")
    
    # Calculate correlations with target
    corr_with_target = data.corrwith(data[target_var]).sort_values(ascending=False)
    st.dataframe(corr_with_target)
    
    # Scatter plot dengan garis regresi
    st.subheader("Scatter Plot dengan Garis Regresi")
    x_var = st.selectbox("Pilih Variabel X:", ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja'])
    fig = px.scatter(data, x=x_var, y=target_var, trendline='ols')
    st.plotly_chart(fig)

    # Source code korelasi
    st.markdown("### Source Code Analisis Korelasi")
    st.code("""
import pandas as pd
import plotly.express as px

# Load data
data = pd.read_csv('data_test_300_kepuasaan_client.csv')

# Correlation matrix
corr_matrix = data.corr()
print(corr_matrix)

# Visualisasi korelasi
fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto")
fig_corr.show()

# Korelasi dengan variabel target
target_var = 'KepuasanKlien'
corr_with_target = data.corrwith(data[target_var]).sort_values(ascending=False)
print(corr_with_target)

# Scatter plot dengan regresi
fig_scatter = px.scatter(data, x='JamKerja', y=target_var, trendline='ols')
fig_scatter.show()
    """)

# === MENU: REGRESI ===
elif selected_menu == "Regresi":
    st.header("📉 Analisis Regresi")
    
    # Pilih tipe regresi
    reg_type = st.selectbox("Pilih Tipe Regresi:", 
                           ["Regresi Linear Sederhana", "Regresi Linear Berganda"])
    
    if reg_type == "Regresi Linear Sederhana":
        st.subheader("Regresi Linear Sederhana")
        x_var = st.selectbox("Pilih Variabel X:", ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja'])
        y_var = 'KepuasanKlien'
        
        # Prepare data
        X = data[x_var]
        X = sm.add_constant(X)  # Add constant term
        y = data[y_var]
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Display results
        st.markdown("**Model Summary:**")
        st.text(model.summary())
        
        # Predictions
        st.subheader("Prediksi")
        new_value = st.slider(f"Masukkan nilai {x_var}:", 
                             int(data[x_var].min()), int(data[x_var].max()), 
                             int(data[x_var].mean()))
        prediction = model.predict([1, new_value])
        st.markdown(f"Prediksi {y_var} untuk {x_var}={new_value}: {prediction[0]:.2f}")
        
        # Plot regression line
        st.subheader("Plot Regresi")
        fig = px.scatter(data, x=x_var, y=y_var, trendline='ols')
        st.plotly_chart(fig)
    
    elif reg_type == "Regresi Linear Berganda":
        st.subheader("Regresi Linear Berganda")
        x_vars = st.multiselect("Pilih Variabel X:", 
                               ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja'])
        y_var = 'KepuasanKlien'
        
        if len(x_vars) > 0:
            # Prepare data
            X = data[x_vars]
            X = sm.add_constant(X)  # Add constant term
            y = data[y_var]
            
            # Fit model
            model = sm.OLS(y, X).fit()
            
            # Display results
            st.markdown("**Model Summary:**")
            st.text(model.summary())
            
            # Uji asumsi
            st.subheader("Uji Asumsi Regresi")
            
            # Normalitas residual
            st.markdown("**1. Normalitas Residual (Shapiro-Wilk Test):**")
            _, p_value = stats.shapiro(model.resid)
            st.markdown(f"p-value: {p_value:.4f}")
            if p_value > 0.05:
                st.markdown("Residual terdistribusi normal (p > 0.05)")
            else:
                st.markdown("Residual tidak terdistribusi normal (p ≤ 0.05)")
            
            # Heteroskedastisitas
            st.markdown("**2. Uji Heteroskedastisitas (Breusch-Pagan Test):**")
            _, p_value, _, _ = sm.stats.het_breuschpagan(model.resid, X)
            st.markdown(f"p-value: {p_value:.4f}")
            if p_value > 0.05:
                st.markdown("Tidak ada heteroskedastisitas (p > 0.05)")
            else:
                st.markdown("Ada heteroskedastisitas (p ≤ 0.05)")
            
            # Multikolinearitas
            st.markdown("**3. Uji Multikolinearitas (VIF):**")
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            st.dataframe(vif_data)
            
            # Prediksi
            st.subheader("Prediksi")
            if len(x_vars) == 1:
                new_value = st.slider(f"Masukkan nilai {x_vars[0]}:", 
                                     int(data[x_vars[0]].min()), int(data[x_vars[0]].max()), 
                                     int(data[x_vars[0]].mean()))
                prediction = model.predict([1, new_value])
                st.markdown(f"Prediksi {y_var} untuk {x_vars[0]}={new_value}: {prediction[0]:.2f}")
            else:
                st.markdown("Untuk prediksi berganda, gunakan kode Python")

    # Source code regresi
    st.markdown("### Source Code Analisis Regresi")
    st.code("""
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px

# Load data
data = pd.read_csv('data_test_300_kepuasaan_client.csv')

# Regresi linear sederhana
X = data['JamKerja']
X = sm.add_constant(X)
y = data['KepuasanKlien']
model = sm.OLS(y, X).fit()
print(model.summary())

# Regresi linear berganda
X = data[['Psichotest', 'JamKerja']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Uji asumsi
# Normalitas residual
_, p_value = stats.shapiro(model.resid)
print(f"Shapiro-Wilk p-value: {p_value}")

# Heteroskedastisitas
_, p_value, _, _ = sm.stats.het_breuschpagan(model.resid, X)
print(f"Breusch-Pagan p-value: {p_value}")

# VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
    """)

# === MENU: ANOVA ===
elif selected_menu == "ANOVA":
    st.header("📊 Analisis Varians (ANOVA)")
    
    # Buat dataset contoh ANOVA
    st.markdown("### Dataset Contoh untuk ANOVA")
    st.markdown("Dataset ini menggambarkan tingkat kepuasan pelanggan berdasarkan jenis layanan dan tingkat pengalaman:")
    
    # Buat dataset contoh
    np.random.seed(42)
    data_anova = pd.DataFrame({
        'Kepuasan': np.concatenate([
            np.random.normal(85, 5, 20),  # Layanan A
            np.random.normal(78, 6, 20),  # Layanan B
            np.random.normal(88, 4, 10)   # Layanan C
        ]),
        'JenisLayanan': ['A']*20 + ['B']*20 + ['C']*10,
        'TingkatPengalaman': np.concatenate([
            ['Pemula']*10 + ['Menengah']*10,  # Layanan A
            ['Pemula']*10 + ['Menengah']*10,  # Layanan B
            ['Ahli']*10                       # Layanan C
        ])
    })
    
    st.dataframe(data_anova.head())
    
    # Pilih tipe ANOVA
    anova_type = st.selectbox("Pilih Tipe ANOVA:", ["One-Way ANOVA", "Two-Way ANOVA"])
    
    if anova_type == "One-Way ANOVA":
        st.subheader("One-Way ANOVA")
        cat_var = st.selectbox("Pilih Variabel Kategori:", ['JenisLayanan'])
        num_var = st.selectbox("Pilih Variabel Numerik:", ['Kepuasan'])
        
        try:
            # Perform ANOVA
            model = ols(f'{num_var} ~ C({cat_var})', data=data_anova).fit()
            anova_table = anova_lm(model)
            
            st.markdown("**Tabel ANOVA:**")
            st.dataframe(anova_table)
            
            # Get p-value safely
            p_value_col = 'PR(>F)' if 'PR(>F)' in anova_table.columns else 'PR(>F)'
            p_value = anova_table[p_value_col][0]
            
            st.markdown(f"**p-value: {p_value:.4f}**")
            if p_value < 0.05:
                st.markdown("✅ Ada perbedaan signifikan antara kelompok (p < 0.05)")
                st.markdown("Kesimpulan: Setidaknya ada satu jenis layanan yang berbeda secara signifikan")
            else:
                st.markdown("❌ Tidak ada perbedaan signifikan antara kelompok (p ≥ 0.05)")
                st.markdown("Kesimpulan: Tidak ada bukti perbedaan antar jenis layanan")
            
            # Post-hoc test (Tukey HSD)
            st.subheader("Post-hoc Test: Tukey HSD")
            try:
                tukey = pairwise_tukeyhsd(endog=data_anova[num_var], groups=data_anova[cat_var], alpha=0.05)
                tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], 
                                      columns=tukey._results_table.data[0])
                st.dataframe(tukey_df)
                
                # Interpretasi Tukey
                significant_pairs = tukey_df[tukey_df['p-adj'] < 0.05]
                if len(significant_pairs) > 0:
                    st.markdown("Pasangan kelompok yang berbeda signifikan:")
                    for _, row in significant_pairs.iterrows():
                        st.markdown(f"- {row['group1']} vs {row['group2']}: p = {row['p-adj']:.4f}")
                else:
                    st.markdown("Tidak ada pasangan kelompok yang berbeda signifikan")
            except Exception as e:
                st.info(f"Post-hoc test error: {str(e)}")
            
            # Visualisasi
            st.subheader("Visualisasi Data")
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.box(data_anova, x=cat_var, y=num_var, title=f'Distribusi {num_var} per {cat_var}')
                st.plotly_chart(fig1)
            
            with col2:
                means = data_anova.groupby(cat_var)[num_var].mean().reset_index()
                fig2 = px.bar(means, x=cat_var, y=num_var, title=f'Rata-rata {num_var} per {cat_var}')
                st.plotly_chart(fig2)
            
            # Descriptive statistics
            st.subheader("Statistik Deskriptif per Kelompok")
            desc_stats = data_anova.groupby(cat_var)[num_var].describe()
            st.dataframe(desc_stats)
            
        except Exception as e:
            st.error(f"Terjadi error saat melakukan ANOVA: {str(e)}")
        
        # Source code
        st.markdown("### Source Code One-Way ANOVA")
        st.code("""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import plotly.express as px

# Buat dataset contoh
np.random.seed(42)
data_anova = pd.DataFrame({
    'Kepuasan': np.concatenate([
        np.random.normal(85, 5, 20),  # Layanan A
        np.random.normal(78, 6, 20),  # Layanan B
        np.random.normal(88, 4, 10)   # Layanan C
    ]),
    'JenisLayanan': ['A']*20 + ['B']*20 + ['C']*10
})

# One-Way ANOVA
model = ols('Kepuasan ~ C(JenisLayanan)', data=data_anova).fit()
anova_table = anova_lm(model)
print(anova_table)

# Post-hoc test
tukey = pairwise_tukeyhsd(endog=data_anova['Kepuasan'], groups=data_anova['JenisLayanan'], alpha=0.05)
print(tukey)

# Visualisasi
fig_box = px.box(data_anova, x='JenisLayanan', y='Kepuasan')
fig_box.show()

fig_bar = px.bar(data_anova.groupby('JenisLayanan')['Kepuasan'].mean().reset_index(), 
                x='JenisLayanan', y='Kepuasan')
fig_bar.show()
        """)
    
    elif anova_type == "Two-Way ANOVA":
        st.subheader("Two-Way ANOVA")
        cat_var1 = st.selectbox("Pilih Variabel Kategori 1:", ['JenisLayanan', 'TingkatPengalaman'])
        cat_var2 = st.selectbox("Pilih Variabel Kategori 2:", ['JenisLayanan', 'TingkatPengalaman'], 
                               index=1 if cat_var1 == 'JenisLayanan' else 0)
        num_var = st.selectbox("Pilih Variabel Numerik:", ['Kepuasan'])
        
        try:
            # Perform Two-Way ANOVA
            model = ols(f'{num_var} ~ C({cat_var1}) + C({cat_var2}) + C({cat_var1}):C({cat_var2})', data=data_anova).fit()
            anova_table = anova_lm(model)
            
            st.markdown("**Tabel ANOVA:**")
            st.dataframe(anova_table)
            
            # Get p-values safely
            p_value_col = 'PR(>F)' if 'PR(>F)' in anova_table.columns else 'PR(>F)'
            
            p_value_main1 = anova_table[p_value_col][0]
            p_value_main2 = anova_table[p_value_col][1]
            p_value_interaction = anova_table[p_value_col][2]
            
            st.markdown(f"**p-value {cat_var1}: {p_value_main1:.4f}**")
            st.markdown(f"**p-value {cat_var2}: {p_value_main2:.4f}**")
            st.markdown(f"**p-value Interaksi: {p_value_interaction:.4f}**")
            
            # Interpretasi
            st.subheader("Interpretasi Hasil")
            
            if p_value_main1 < 0.05:
                st.markdown(f"✅ Efek utama {cat_var1} signifikan (p < 0.05)")
            else:
                st.markdown(f"❌ Efek utama {cat_var1} tidak signifikan (p ≥ 0.05)")
                
            if p_value_main2 < 0.05:
                st.markdown(f"✅ Efek utama {cat_var2} signifikan (p < 0.05)")
            else:
                st.markdown(f"❌ Efek utama {cat_var2} tidak signifikan (p ≥ 0.05)")
                
            if p_value_interaction < 0.05:
                st.markdown("✅ Ada interaksi signifikan antara faktor (p < 0.05)")
                st.markdown("Kesimpulan: Efek satu faktor bergantung pada faktor lain")
                st.markdown("Contoh: Perbedaan kepuasan antar layanan berbeda untuk setiap tingkat pengalaman")
            else:
                st.markdown("❌ Tidak ada interaksi signifikan antara faktor (p ≥ 0.05)")
                st.markdown("Kesimpulan: Efek faktor bersifat independen")
            
            # Visualisasi interaksi
            st.subheader("Visualisasi Interaksi")
            interaction_plot = data_anova.groupby([cat_var1, cat_var2])[num_var].mean().reset_index()
            fig = px.line(interaction_plot, x=cat_var1, y=num_var, color=cat_var2, 
                        title=f'Interaksi {cat_var1} x {cat_var2} terhadap {num_var}')
            st.plotly_chart(fig)
            
            # Simple effects analysis
            st.subheader("Analisis Efek Sederhana")
            if cat_var1 == 'JenisLayanan' and cat_var2 == 'TingkatPengalaman':
                for tingkat in data_anova['TingkatPengalaman'].unique():
                    subset = data_anova[data_anova['TingkatPengalaman'] == tingkat]
                    model_simple = ols('Kepuasan ~ C(JenisLayanan)', data=subset).fit()
                    anova_simple = anova_lm(model_simple)
                    p_simple = anova_simple['PR(>F)'][0]
                    st.markdown(f"**Efek JenisLayanan untuk {tingkat}: p = {p_simple:.4f}**")
            
        except Exception as e:
            st.error(f"Terjadi error saat melakukan Two-Way ANOVA: {str(e)}")
        
        # Source code
        st.markdown("### Source Code Two-Way ANOVA")
        st.code("""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import plotly.express as px

# Buat dataset contoh
np.random.seed(42)
data_anova = pd.DataFrame({
    'Kepuasan': np.concatenate([
        np.random.normal(85, 5, 20),  # Layanan A
        np.random.normal(78, 6, 20),  # Layanan B
        np.random.normal(88, 4, 10)   # Layanan C
    ]),
    'JenisLayanan': ['A']*20 + ['B']*20 + ['C']*10,
    'TingkatPengalaman': np.concatenate([
        ['Pemula']*10 + ['Menengah']*10,  # Layanan A
        ['Pemula']*10 + ['Menengah']*10,  # Layanan B
        ['Ahli']*10                       # Layanan C
    ])
})

# Two-Way ANOVA
model = ols('Kepuasan ~ C(JenisLayanan) + C(TingkatPengalaman) + C(JenisLayanan):C(TingkatPengalaman)', 
            data=data_anova).fit()
anova_table = anova_lm(model)
print(anova_table)

# Visualisasi interaksi
interaction_plot = data_anova.groupby(['JenisLayanan', 'TingkatPengalaman'])['Kepuasan'].mean().reset_index()
fig = px.line(interaction_plot, x='JenisLayanan', y='Kepuasan', color='TingkatPengalaman')
fig.show()

# Analisis efek sederhana
for tingkat in data_anova['TingkatPengalaman'].unique():
    subset = data_anova[data_anova['TingkatPengalaman'] == tingkat]
    model_simple = ols('Kepuasan ~ C(JenisLayanan)', data=subset).fit()
    anova_simple = anova_lm(model_simple)
    p_simple = anova_simple['PR(>F)'][0]
    print(f"Effek JenisLayanan untuk {tingkat}: p = {p_simple}")
        """)

# === MENU: UJI HIPOTESIS ===
elif selected_menu == "Uji Hipotesis":
    st.header("🧪 Uji Hipotesis")
    
    # Pilih tipe uji
    test_type = st.selectbox("Pilih Tipe Uji:", 
                            ["Uji-t Sampel Tunggal", "Uji-t Sampel Berpasangan", "Uji-t Sampel Independen"])
    
    if test_type == "Uji-t Sampel Tunggal":
        st.subheader("Uji-t Sampel Tunggal")
        var = st.selectbox("Pilih Variabel:", ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja', 'KepuasanKlien'])
        test_value = st.number_input("Nilai Hipotesis (μ₀):", value=50.0)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(data[var], test_value)
        
        st.markdown(f"**Statistik t: {t_stat:.4f}**")
        st.markdown(f"**p-value: {p_value:.4f}**")
        
        if p_value < 0.05:
            st.markdown("✅ Tolak H₀ (p < 0.05)")
            st.markdown("Kesimpulan: Rata-rata berbeda signifikan dari hipotesis")
        else:
            st.markdown("❌ Gagal tolak H₀ (p ≥ 0.05)")
            st.markdown("Kesimpulan: Tidak ada bukti rata-rata berbeda dari hipotesis")
    
    elif test_type == "Uji-t Sampel Berpasangan":
        st.subheader("Uji-t Sampel Berpasangan")
        var1 = st.selectbox("Pilih Variabel 1:", ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja', 'KepuasanKlien'])
        var2 = st.selectbox("Pilih Variabel 2:", ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja', 'KepuasanKlien'])
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(data[var1], data[var2])
        
        st.markdown(f"**Statistik t: {t_stat:.4f}**")
        st.markdown(f"**p-value: {p_value:.4f}**")
        
        if p_value < 0.05:
            st.markdown("✅ Tolak H₀ (p < 0.05)")
            st.markdown("Kesimpulan: Terdapat perbedaan signifikan antara dua variabel")
        else:
            st.markdown("❌ Gagal tolak H₀ (p ≥ 0.05)")
            st.markdown("Kesimpulan: Tidak ada perbedaan signifikan antara dua variabel")
    
    elif test_type == "Uji-t Sampel Independen":
        st.subheader("Uji-t Sampel Independen")
        var = st.selectbox("Pilih Variabel:", ['Psichotest', 'LamaPend', 'IQ', 'JamTraining', 'JamKerja', 'KepuasanKlien'])
        group_var = st.selectbox("Pilih Variabel Kelompok:", ['LamaPend'])
        
        # Create groups
        group1 = data[data[group_var] == 0][var]
        group2 = data[data[group_var] == 1][var]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        st.markdown(f"**Statistik t: {t_stat:.4f}**")
        st.markdown(f"**p-value: {p_value:.4f}**")
        
        if p_value < 0.05:
            st.markdown("✅ Tolak H₀ (p < 0.05)")
            st.markdown("Kesimpulan: Terdapat perbedaan signifikan antar kelompok")
        else:
            st.markdown("❌ Gagal tolak H₀ (p ≥ 0.05)")
            st.markdown("Kesimpulan: Tidak ada perbedaan signifikan antar kelompok")
    
    # Jenis Kesalahan
    st.markdown("---")
    st.markdown("**Jenis Kesalahan dalam Uji Hipotesis:**")
    st.markdown("- **Type I Error (α):** Menolak H₀ padahal H₀ benar (False Positive)")
    st.markdown("- **Type II Error (β):** Gagal menolak H₀ padahal H₀ salah (False Positive)")

    # Source code uji hipotesis
    st.markdown("### Source Code Uji Hipotesis")
    st.code("""
import pandas as pd
import scipy.stats as stats

# Load data
data = pd.read_csv('data_test_300_kepuasan_client.csv')

# Uji-t sampel tunggal
t_stat, p_value = stats.ttest_1samp(data['KepuasanKlien'], 50)
print(f"t-statistic: {t_stat}, p-value: {p_value}")

# Uji-t sampel berpasangan
t_stat, p_value = stats.ttest_rel(data['Psichotest'], data['KepuasanKlien'])
print(f"t-statistic: {t_stat}, p-value: {p_value}")

# Uji-t sampel independen
group1 = data[data['LamaPend'] == 0]['KepuasanKlien']
group2 = data[data['LamaPend'] == 1]['KepuasanKlien']
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"t-statistic: {t_stat}, p-value: {p_value}")
    """)
