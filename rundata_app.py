import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from factor_analyzer import FactorAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -----------------------------------------------------------------------------
# ページ設定
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RunData Analyzer",
    page_icon="🏃‍♂️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# セッションステート初期化
# -----------------------------------------------------------------------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []  # [(file_name, dataframe)]

# -----------------------------------------------------------------------------
# アップロード & データ選択ユーティリティ
# -----------------------------------------------------------------------------

def upload_data():
    """CSV アップロードとバリデーション"""
    uploaded_file = st.file_uploader("CSV ファイルを選択", type=["csv"])
    if uploaded_file is None:
        return None

    # ▼ CSV 読み込みを安全に行う
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.toast("⚠️ このファイルは使用できません。CSV 形式を確認してください。", icon="⚠️")
        st.stop()  # ページレンダリングを中断

    # ▼ 履歴に追加（重複除外・最大5件）
    st.session_state.uploaded_files.insert(0, (uploaded_file.name, df))
    st.session_state.uploaded_files = st.session_state.uploaded_files[:5]
    return df


def pick_from_history():
    """履歴から DataFrame を選択"""
    if not st.session_state.uploaded_files:
        st.info("まず CSV をアップロードしてください。")
        return None

    names = [f"{i+1}. {name}" for i, (name, _) in enumerate(st.session_state.uploaded_files)]
    idx = st.selectbox("アップロード履歴", range(len(names)), format_func=lambda i: names[i])
    return st.session_state.uploaded_files[idx][1]

# -----------------------------------------------------------------------------
# サイドバー：ページ切替
# -----------------------------------------------------------------------------
page = st.sidebar.radio("ページを選択", (
    "データ取込", "EDA", "統計解析", "可視化"
))

# -----------------------------------------------------------------------------
# ページ 1: データ取込
# -----------------------------------------------------------------------------
if page == "データ取込":
    st.header("📂 データ取込")
    df = upload_data()
    if df is not None:
        st.success(f"{df.shape[0]} 行 × {df.shape[1]} 列 を読み込みました。")
        st.dataframe(df.head())

# -----------------------------------------------------------------------------
# ページ 2: EDA
# -----------------------------------------------------------------------------
elif page == "EDA":
    st.header("🔍 Exploratory Data Analysis")
    df = pick_from_history()
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("基本統計量")
            st.dataframe(df.describe(include="all"))
        with col2:
            st.subheader("欠損割合")
            na_rate = df.isna().mean().to_frame("NaN %").sort_values("NaN %", ascending=False)
            st.dataframe((na_rate * 100).round(1))
        st.subheader("相関ヒートマップ")
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            fig = px.imshow(num_df.corr(), text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("数値列がありません。")

# -----------------------------------------------------------------------------
# ページ 3: 統計解析
# -----------------------------------------------------------------------------
elif page == "統計解析":
    st.header("📊 統計解析")
    df = pick_from_history()
    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 3:
            st.warning("因子分析には数値列が最低 3 列必要です。")
        else:
            st.subheader("因子分析")
            n_factors = st.slider("抽出因子数", 1, min(10, len(numeric_cols)), 2)
            fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
            fa.fit(df[numeric_cols].dropna())
            loadings = pd.DataFrame(
                fa.loadings_,
                index=numeric_cols,
                columns=[f"Factor{i+1}" for i in range(n_factors)],
            )
            st.dataframe(loadings.round(3))

        # 簡易回帰モデル
        st.subheader("線形回帰")
        target = st.selectbox("目的変数", numeric_cols)
        features = st.multiselect("説明変数", [c for c in numeric_cols if c != target])
        if target and features:
            X = df[features].dropna()
            y = df.loc[X.index, target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.metric("R2", f"{r2_score(y_test, y_pred):.3f}")

            coef_df = pd.DataFrame({"Feature": features, "Coef": model.coef_})
            st.dataframe(coef_df)

# -----------------------------------------------------------------------------
# ページ 4: 可視化
# -----------------------------------------------------------------------------
elif page == "可視化":
    st.header("📈 可視化")
    df = pick_from_history()
    if df is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        col_x = st.selectbox("X 軸", num_cols)
        col_y = st.selectbox("Y 軸", num_cols, index=1 if len(num_cols) > 1 else 0)
        if col_x and col_y:
            fig = px.scatter(df, x=col_x, y=col_y, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# ダウンロード
# -----------------------------------------------------------------------------
if page != "データ取込" and "uploaded_files" in st.session_state and st.session_state.uploaded_files:
    name, last_df = st.session_state.uploaded_files[0]
    csv = last_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 最新データを CSV でダウンロード", csv, file_name=f"{Path(name).stem}_processed.csv", mime="text/csv")
