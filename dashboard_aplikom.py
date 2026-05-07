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
st.dataframe(data.head())

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

# === MENU: ANOVA ===
elif selected_menu == "ANOVA":
    st.header("📊 Analisis Varians (ANOVA)")
    
    # Buat dataset contoh ANOVA
    st.markdown("### Dataset Contoh untuk ANOVA")
    st.markdown("Dataset ini menggambarkan tingkat kepuasan pelanggan berdasarkan jenis layanan dan tingkat pengalaman:")
    
    # Buat dataset contoh yang lebih seimbang
    np.random.seed(42)
    data_anova = pd.DataFrame({
        'Kepuasan': np.concatenate([
            np.random.normal(85, 5, 15),   # Layanan A - Pemula
            np.random.normal(88, 4, 15),   # Layanan A - Menengah
            np.random.normal(78, 6, 15),   # Layanan B - Pemula
            np.random.normal(82, 5, 15),   # Layanan B - Menengah
            np.random.normal(90, 4, 15),   # Layanan C - Pemula
            np.random.normal(92, 3, 15),   # Layanan C - Menengah
        ]),
        'JenisLayanan': ['A']*30 + ['B']*30 + ['C']*30,
        'TingkatPengalaman': ['Pemula']*15 + ['Menengah']*15 + ['Pemula']*15 + ['Menengah']*15 + ['Pemula']*15 + ['Menengah']*15
    })
    
    st.dataframe(data_anova.head())
    st.write(f"Total observasi: {len(data_anova)}")
    
    # Tampilkan tabel kontingensi
    contingency_table = pd.crosstab(data_anova['JenisLayanan'], data_anova['TingkatPengalaman'])
    st.markdown("**Tabel Kontingensi:**")
    st.dataframe(contingency_table)
    
    # Pilih tipe ANOVA
    anova_type = st.selectbox("Pilih Tipe ANOVA:", ["One-Way ANOVA", "Two-Way ANOVA"])
    
    if anova_type == "One-Way ANOVA":
        st.subheader("One-Way ANOVA")
        cat_var = st.selectbox("Pilih Variabel Kategori:", ['JenisLayanan', 'TingkatPengalaman'])
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
                st.markdown("Kesimpulan: Setidaknya ada satu kelompok yang berbeda secara signifikan")
            else:
                st.markdown("❌ Tidak ada perbedaan signifikan antara kelompok (p ≥ 0.05)")
                st.markdown("Kesimpulan: Tidak ada bukti perbedaan antar kelompok")
            
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
            st.error(f"Terjadi error saat melakukan One-Way ANOVA: {str(e)}")
        
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
        np.random.normal(85, 5, 15),   # Layanan A
        np.random.normal(78, 6, 15),   # Layanan B
        np.random.normal(90, 4, 15)    # Layanan C
    ]),
    'JenisLayanan': ['A']*15 + ['B']*15 + ['C']*15
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
        
        # Pilih variabel dengan validasi
        cat_var1 = st.selectbox("Pilih Variabel Kategori 1:", ['JenisLayanan', 'TingkatPengalaman'])
        cat_var2 = st.selectbox("Pilih Variabel Kategori 2:", ['JenisLayanan', 'TingkatPengalaman'], 
                               index=1 if cat_var1 == 'JenisLayanan' else 0)
        num_var = st.selectbox("Pilih Variabel Numerik:", ['Kepuasan'])
        
        # Validasi
        if cat_var1 == cat_var2:
            st.error("❌ Pilih dua variabel kategori yang berbeda!")
            st.stop()
        
        try:
            # Cek frekuensi sel
            contingency_table = pd.crosstab(data_anova[cat_var1], data_anova[cat_var2])
            st.markdown("**Tabel Kontingensi (frekuensi):**")
            st.dataframe(contingency_table)
            
            # Cek apakah ada sel dengan frekuensi < 5
            if (contingency_table < 5).any().any():
                st.warning("⚠️ Beberapa sel memiliki frekuensi < 5. Hasil ANOVA mungkin tidak optimal.")
            
            # Perform Two-Way ANOVA
            formula = f'{num_var} ~ C({cat_var1}) + C({cat_var2}) + C({cat_var1}):C({cat_var2})'
            st.markdown(f"**Rumus ANOVA:** `{formula}`")
            
            model = ols(formula, data=data_anova).fit()
            anova_table = anova_lm(model)
            
            st.markdown("**Tabel ANOVA:**")
            st.dataframe(anova_table)
            
            # Get p-values safely
            p_value_col = 'PR(>F)' if 'PR(>F)' in anova_table.columns else 'PR(>F)'
            
            # Ambil nilai p-value dengan penanganan error
            try:
                p_value_main1 = anova_table[p_value_col][0]
                p_value_main2 = anova_table[p_value_col][1]
                p_value_interaction = anova_table[p_value_col][2]
            except IndexError:
                st.error("❌ Tidak cukup data untuk menghitung semua efek. Dataset perlu diperbesar.")
                st.stop()
            
            st.markdown(f"**p-value {cat_var1}: {p_value_main1:.4f}**")
            st.markdown(f"**p-value {cat_var2}: {p_value_main2:.4f}**")
            st.markdown(f"**p-value Interaksi: {p_value_interaction:.4f}**")
            
            # Interpretasi
            st.subheader("📊 Interpretasi Hasil")
            
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
                st.markdown("📝 Kesimpulan: Efek satu faktor bergantung pada faktor lain")
                
                # Visualisasi interaksi
                st.subheader("📈 Visualisasi Interaksi")
                interaction_plot = data_anova.groupby([cat_var1, cat_var2])[num_var].mean().reset_index()
                fig = px.line(interaction_plot, x=cat_var1, y=num_var, color=cat_var2, 
                            title=f'Interaksi {cat_var1} x {cat_var2} terhadap {num_var}')
                st.plotly_chart(fig)
                
                # Analisis simple effects
                st.subheader("🔍 Analisis Efek Sederhana")
                for level in data_anova[cat_var2].unique():
                    subset = data_anova[data_anova[cat_var2] == level]
                    if len(subset) > 10:  # Pastikan cukup data
                        model_simple = ols(f'{num_var} ~ C({cat_var1})', data=subset).fit()
                        anova_simple = anova_lm(model_simple)
                        try:
                            p_simple = anova_simple['PR(>F)'][0]
                            st.markdown(f"**Efek {cat_var1} untuk {cat_var2}={level}: p = {p_simple:.4f}**")
                            
                            # Visualisasi simple effects
                            fig_simple = px.box(subset, x=cat_var1, y=num_var, 
                                               title=f'Distribusi {num_var} per {cat_var1} ({cat_var2}={level})')
                            st.plotly_chart(fig_simple)
                        except:
                            st.info(f"Tidak bisa menghitung efek sederhana untuk {cat_var2}={level}")
            else:
                st.markdown("❌ Tidak ada interaksi signifikan antara faktor (p ≥ 0.05)")
                st.markdown("📝 Kesimpulan: Efek faktor bersifat independen")
                
                # Visualisasi tanpa interaksi
                st.subheader("📊 Visualisasi Efek Utama")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.box(data_anova, x=cat_var1, y=num_var, title=f'Distribusi per {cat_var1}')
                    st.plotly_chart(fig1)
                
                with col2:
                    fig2 = px.box(data_anova, x=cat_var2, y=num_var, title=f'Distribusi per {cat_var2}')
                    st.plotly_chart(fig2)
            
        except Exception as e:
            st.error(f"❌ Terjadi error saat melakukan Two-Way ANOVA: {str(e)}")
            st.markdown("### 🔧 Kemungkinan Penyebab dan Solusi:")
            st.markdown("1. **Data tidak cukup**: Pastikan setiap kombinasi kategori memiliki minimal 5 observasi")
            st.markdown("2. **Variabel sama**: Pastikan memilih dua variabel kategori yang berbeda")
            st.markdown("3. **Error statistik**: Coba dataset yang lebih besar atau seimbang")
            
            # Alternative approach dengan pingouin
            try:
                st.markdown("### 🔄 Alternatif: Menggunakan Pingouin")
                import pingouin as pg
                aov = pg.anova(data=data_anova, dv=num_var, between=[cat_var1, cat_var2])
                st.dataframe(aov)
            except:
                st.info("Pingouin tidak tersedia atau gagal dijalankan")
            
            # Source code
            st.markdown("### 💻 Source Code Two-Way ANOVA")
            st.code("""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import plotly.express as px

# Buat dataset contoh yang seimbang
np.random.seed(42)
data_anova = pd.DataFrame({
    'Kepuasan': np.concatenate([
        np.random.normal(85, 5, 15),   # Layanan A - Pemula
        np.random.normal(88, 4, 15),   # Layanan A - Menengah
        np.random.normal(78, 6, 15),   # Layanan B - Pemula
        np.random.normal(82, 5, 15),   # Layanan B - Menengah
        np.random.normal(90, 4, 15),   # Layanan C - Pemula
        np.random.normal(92, 3, 15),   # Layanan C - Menengah
    ]),
    'JenisLayanan': ['A']*30 + ['B']*30 + ['C']*30,
    'TingkatPengalaman': ['Pemula']*15 + ['Menengah']*15 + ['Pemula']*15 + ['Menengah']*15 + ['Pemula']*15 + ['Menengah']*15
})

# Cek tabel kontingensi
contingency_table = pd.crosstab(data_anova['JenisLayanan'], data_anova['TingkatPengalaman'])
print("Tabel Kontingensi:")
print(contingency_table)

# Two-Way ANOVA
model = ols('Kepuasan ~ C(JenisLayanan) + C(TingkatPengalaman) + C(JenisLayanan):C(TingkatPengalaman)', 
            data=data_anova).fit()
anova_table = anova_lm(model)
print("\\nTabel ANOVA:")
print(anova_table)

# Visualisasi interaksi
interaction_plot = data_anova.groupby(['JenisLayanan', 'TingkatPengalaman'])['Kepuasan'].mean().reset_index()
fig = px.line(interaction_plot, x='JenisLayanan', y='Kepuasan', color='TingkatPengalaman')
fig.show()
            """)
