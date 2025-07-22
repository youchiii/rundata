import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import FactorAnalysis

st.set_page_config(page_title="時系列データ解析アプリ", layout="wide")
st.title("🏃‍♂️ マラソンデータ解析アプリ")
st.write("CSVをアップロードし、可視化・統計解析・因子分析・重回帰分析を行えます。")

df = None
uploaded_file = st.sidebar.file_uploader("📂 CSVファイルをアップロード", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding="shift_jis")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="utf-8")

page = st.sidebar.radio("📄 表示ページを選択", ["📈 データ可視化", "📊 統計解析", "📉 重回帰分析", "🧠 因子分析"])

if df is not None:
    numeric_cols = df.select_dtypes(include='number').columns
    cat_cols = [col for col in df.columns if df[col].nunique() < 10]

    if page == "📈 データ可視化":
        st.subheader("📋 データプレビュー")
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 各項目の可視化・統計")
        for col in numeric_cols:
            st.markdown(f"### 🔹 {col}")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.plotly_chart(px.line(df, y=col, title=f"{col} の推移", markers=True), use_container_width=True)
                st.plotly_chart(px.histogram(df, x=col, title=f"{col} の度数分布"), use_container_width=True)
            with col2:
                st.markdown("**基本統計量**")
                st.write(f"- 平均: {df[col].mean():.2f}")
                st.write(f"- 中央値: {df[col].median():.2f}")
                mode_val = df[col].mode().iat[0] if not df[col].mode().empty else "なし"
                st.write(f"- 最頻値: {mode_val}")
                st.write(f"- 分散: {df[col].var():.2f}")
                st.write(f"- 標準偏差: {df[col].std():.2f}")

        st.subheader("📊 相関係数マトリクス")
        st.plotly_chart(px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale="Blues"), use_container_width=True)

    elif page == "📊 統計解析":
        st.subheader("🔁 クロス集計・χ²検定")
        if len(cat_cols) >= 2:
            cat1 = st.selectbox("カテゴリ列①", cat_cols, key="cat1")
            cat2 = st.selectbox("カテゴリ列②", cat_cols, key="cat2")

            if st.button("χ²検定を実行"):
                ctab = pd.crosstab(df[cat1], df[cat2])
                st.dataframe(ctab)

                chi2, p, dof, expected = chi2_contingency(ctab)
                st.write(f"χ²統計量: {chi2:.4f}, 自由度: {dof}, p値: {p:.4f}")
                st.success("有意な差あり（p < 0.05）" if p < 0.05 else "有意な差なし")

        st.subheader("📐 t検定（2群）")
        if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
            num_col = st.selectbox("数値列", numeric_cols)
            group_col = st.selectbox("グループ列", cat_cols)

            groups = df[group_col].dropna().unique()
            if len(groups) == 2:
                g1, g2 = groups
                data1 = df[df[group_col] == g1][num_col]
                data2 = df[df[group_col] == g2][num_col]

                t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
                st.write(f"{g1} vs {g2}")
                st.write(f"t統計量: {t_stat:.4f}, p値: {p_val:.4f}")
                st.success("有意差あり（p < 0.05）" if p_val < 0.05 else "有意差なし")

    elif page == "📉 重回帰分析":
        if len(numeric_cols) >= 2:
            st.subheader("📉 重回帰分析")
            target = st.selectbox("🎯 目的変数", numeric_cols)
            features = st.multiselect("🧮 説明変数", [col for col in numeric_cols if col != target])

            if features:
                X = df[features].dropna()
                y = df[target].loc[X.index]

                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                st.markdown("#### ✅ 結果")
                st.write(f"決定係数（R²）: {r2_score(y, y_pred):.4f}")
                st.write(f"平均二乗誤差（MSE）: {mean_squared_error(y, y_pred):.4f}")

                coef_df = pd.DataFrame({"変数": features, "係数": model.coef_})
                st.dataframe(coef_df)

                equation = f"{target} = " + " + ".join([f"{coef:.2f}×{var}" for coef, var in zip(model.coef_, features)])
                st.markdown(f"#### 📏 回帰式：{equation}")

                st.markdown("#### 実測 vs 予測")
                fig = px.scatter(x=y, y=y_pred, labels={"x": "実測値", "y": "予測値"})
                fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(dash="dash"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("説明変数を1つ以上選択してください。")

    elif page == "🧠 因子分析":
        st.subheader("🧠 因子分析（Factor Analysis）")
        n_factors = st.slider("抽出する因子数", 1, min(len(numeric_cols), 10), 2)
        st.markdown(
    """
    ℹ️ **因子数のヒント**  
    - 因子数は「相関のある変数群をいくつの潜在的な要因（因子）で説明できるか」の目安です。  
    - 通常、**固有値 > 1** の因子数や、「見たい視点」に応じて2～5個程度を選ぶことが多いです。  
    - 変数数が少ない場合は、**因子数を少なめ**にするのがおすすめです。
    """
)
        if len(numeric_cols) >= 2:
            fa = FactorAnalysis(n_components=n_factors)
            fa.fit(df[numeric_cols].dropna())

            st.write("🔢 固有値（各因子の寄与）")
            evr = np.var(fa.transform(df[numeric_cols].dropna()), axis=0)
            for i, val in enumerate(evr):
                st.write(f"因子{i+1}: {val:.4f}")

            st.write("📊 因子負荷量（Factor Loadings）")
            loadings = pd.DataFrame(fa.components_.T, index=numeric_cols, columns=[f"因子{i+1}" for i in range(n_factors)])
            st.dataframe(loadings.style.highlight_max(axis=1))

else:
    st.info("左のサイドバーからCSVファイルをアップロードしてください。")
