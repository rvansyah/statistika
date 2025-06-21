import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

st.title("Aplikasi Statistika Parametrik & Non-Parametrik")

st.header("Input Data")
data_input = st.text_area("Masukkan data numerik (pisahkan dengan koma atau spasi):", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
data = []
for item in data_input.replace(',', ' ').split():
    try:
        data.append(float(item))
    except ValueError:
        pass

if data:
    st.subheader("Statistika Parametrik")
    st.write(f"Mean: {np.mean(data):.3f}")
    st.write(f"Standar Deviasi: {np.std(data, ddof=1):.3f}")
    st.write(f"Range: {np.ptp(data):.3f}")

    st.subheader("Statistika Non-Parametrik")
    st.write(f"Median: {np.median(data):.3f}")
    st.write(f"Modus: {stats.mode(data, keepdims=True)[0][0]:.3f}")

st.header("Uji Dua Kelompok")
col1, col2 = st.columns(2)
with col1:
    group1_input = st.text_area("Kelompok 1", "1, 2, 3, 4, 5")
with col2:
    group2_input = st.text_area("Kelompok 2", "6, 7, 8, 9, 10")

def parse_group_input(text):
    group = []
    for item in text.replace(',', ' ').split():
        try:
            group.append(float(item))
        except ValueError:
            pass
    return group

group1 = parse_group_input(group1_input)
group2 = parse_group_input(group2_input)

if group1 and group2:
    st.write("### Uji t-test Independen (Parametrik)")
    t_stat, t_p = stats.ttest_ind(group1, group2, equal_var=False)
    st.write(f"t-statistik: {t_stat:.3f}, p-value: {t_p:.3f}")

    st.write("### Uji Mann-Whitney U (Non-Parametrik)")
    u_stat, u_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    st.write(f"U-statistik: {u_stat:.3f}, p-value: {u_p:.3f}")

st.header("Uji ANOVA (Analisis Varians Satu Arah)")
anova_input = st.text_area("Masukkan data beberapa kelompok untuk ANOVA (pisahkan kelompok dengan baris baru, data dalam kelompok pisahkan koma):", "1,2,3\n4,5,6\n7,8,9")
anova_groups = []
for line in anova_input.strip().split('\n'):
    group = []
    for item in line.replace(',', ' ').split():
        try:
            group.append(float(item))
        except ValueError:
            pass
    if group:
        anova_groups.append(group)

if len(anova_groups) > 1:
    f_stat, p_value = stats.f_oneway(*anova_groups)
    st.write(f"F-statistik: {f_stat:.3f}, p-value: {p_value:.3f}")

st.header("Uji MANOVA (Analisis Varians Multivariat)")
st.markdown(
    "Masukkan data MANOVA dalam format tabel (kolom dipisah koma, baris dipisah baris baru), kolom pertama variabel kelompok, sisanya variabel dependen. Contoh:<br>"
    "`A,1,2`\n`A,2,3`\n`B,4,5`\n`B,5,6`",
    unsafe_allow_html=True
)
manova_input = st.text_area("Data MANOVA", "A,1,2\nA,2,3\nB,4,5\nB,5,6")

manova_data = []
for line in manova_input.strip().split('\n'):
    row = [x.strip() for x in line.split(',')]
    if len(row) >= 3:
        manova_data.append(row)

if manova_data:
    df = pd.DataFrame(manova_data)
    group_col = df.columns[0]
    dep_cols = df.columns[1:]
    try:
        df[dep_cols] = df[dep_cols].astype(float)
        formula = ' + '.join([f'{col}' for col in dep_cols])
        manova = MANOVA.from_formula(f'{formula} ~ {group_col}', data=df)
        st.text(manova.mv_test())
    except Exception as e:
        st.warning(f"Data MANOVA tidak valid: {e}")

st.info("Aplikasi ini menggunakan Streamlit, NumPy, SciPy, pandas, dan statsmodels. "
        "Untuk menjalankan: `pip install streamlit numpy scipy pandas statsmodels` lalu `streamlit run statistika_app.py`")