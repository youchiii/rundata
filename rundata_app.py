import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency, ttest_ind

st.set_page_config(page_title="時系列データ解析アプリ", layout="wide")

st.title("🏃‍♂️ マラソンデータ解析アプリ")
st.write("CSVをアップロードし、可視化・統計解析を実施できます。")

# ファイルアップロード
df = None
uploaded_file = st.sidebar.file_uploader("📂 CSVファイルをアップロード", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding="shift_jis")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="utf-8")

# ページ切り替え
page = st.sidebar.radio("📄 表示ページを選択", ["📈 データ可視化", "📊 統計解析"])

if df is not None:
    numeric_cols = df.select_dtypes(include='number').columns
    # object型 + ユニーク数が少ない列をカテゴリとみなす
    cat_cols = [col for col in df.columns if df[col].nunique() < 10]

    if page == "📈 データ可視化":
        st.subheader("📋 データプレビュー")
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 各項目の可視化・統計")
        for col in numeric_cols:
            st.markdown(f"### 🔹 {col}")
            col1, col2 = st.columns([2, 1])

            with col1:
                fig_line = px.line(df, y=col, title=f"{col} の推移", markers=True, template="plotly_white")
                fig_line.update_layout(xaxis_title="時間（分）", yaxis_title=col)
                st.plotly_chart(fig_line, use_container_width=True)

                fig_hist = px.histogram(df, x=col, title=f"{col} の度数分布", template="plotly_white")
                fig_hist.update_layout(xaxis_title=col, yaxis_title="度数", bargap=0.2)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                st.markdown("**基本統計量**")
                st.write(f"- 平均: {df[col].mean():.2f}")
                st.write(f"- 中央値: {df[col].median():.2f}")
                mode_val = df[col].mode().iat[0] if not df[col].mode().empty else "なし"
                st.write(f"- 最頻値: {mode_val}")
                st.write(f"- 分散: {df[col].var():.2f}")
                st.write(f"- 標準偏差: {df[col].std():.2f}")

        st.subheader("📊 相関係数マトリクス")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", title="数値項目の相関係数")
        st.plotly_chart(fig_corr, use_container_width=True)

    elif page == "📊 統計解析":
        st.header("📊 統計解析ページ")

        st.subheader("🔁 クロス集計・χ²検定")
        if len(cat_cols) >= 2:
            cat1 = st.selectbox("カテゴリ列①", cat_cols, key="cat1")
            cat2 = st.selectbox("カテゴリ列②", cat_cols, key="cat2")

            if st.button("クロス集計とχ²検定を実行"):
                ctab = pd.crosstab(df[cat1], df[cat2])
                st.dataframe(ctab)

                chi2, p, dof, expected = chi2_contingency(ctab)
                st.write(f"χ²統計量: {chi2:.4f}, 自由度: {dof}, p値: {p:.4f}")
                if p < 0.05:
                    st.success("有意な差があります（p < 0.05）")
                else:
                    st.info("有意な差は見られません")
        else:
            st.info("カテゴリ列が2つ以上必要です")

        st.subheader("📐 t検定（2群）")
        if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
            num_col = st.selectbox("数値列", numeric_cols, key="num_col")
            group_col = st.selectbox("グループ列（カテゴリ）", cat_cols, key="group_col")

            groups = df[group_col].dropna().unique()
            st.write(f"グループの種類: {groups}")

            if len(groups) == 2:
                g1, g2 = groups
                data1 = df[df[group_col] == g1][num_col].dropna()
                data2 = df[df[group_col] == g2][num_col].dropna()

                t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
                st.write(f"グループ: {g1} vs {g2}")
                st.write(f"t統計量: {t_stat:.4f}, p値: {p_val:.4f}")
                if p_val < 0.05:
                    st.success("有意差あり（p < 0.05）")
                else:
                    st.info("有意差なし")
            else:
                st.warning("グループは2つにしてください")
        else:
            st.info("カテゴリ列と数値列が必要です")
else:
    st.info("左のサイドバーからCSVファイルをアップロードしてください。")