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
# ページ設定 & テーマカラー（.streamlit/config.toml を使う場合はファイル側で指定）
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RunData Analyzer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏃‍♂️ RunData Analyzer")

# -----------------------------------------------------------------------------
# セッションステート初期化
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # 最後にアップロードした DataFrame を最大5件保存
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

# -----------------------------------------------------------------------------
# サイドバー: ページ切替 & ファイルアップロード
# -----------------------------------------------------------------------------
page = st.sidebar.radio("ページを選択", [
    "データ取込",
    "EDA",
    "統計解析",
    "可視化"
])

uploaded_file = st.sidebar.file_uploader("📂 CSV ファイルをアップロード", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.current_df = df
        # 履歴の先頭に追加、同名ファイルは重複排除
        st.session_state.history = [h for h in st.session_state.history if h[0] != uploaded_file.name]
        st.session_state.history.insert(0, (uploaded_file.name, df))
        st.session_state.history = st.session_state.history[:5]  # 最大5件
        st.toast(f"{uploaded_file.name} を読み込みました", icon="✅")
    except Exception as e:
        st.toast(f"読み込みエラー: {e}", icon="❌")

# アップロード履歴から再選択
if st.session_state.history:
    names = [h[0] for h in st.session_state.history]
    choice = st.sidebar.selectbox("履歴から選択", names, index=0 if st.session_state.current_df is None else names.index(st.session_state.history[0][0]))
    st.session_state.current_df = dict(st.session_state.history)[choice]

# データがない場合はメインにヒント表示して早期 return
if st.session_state.current_df is None:
    st.info("左のサイドバーから CSV をアップロードしてください。")
    st.stop()

# DataFrame ショートハンド
_df = st.session_state.current_df

# -----------------------------------------------------------------------------
# 1. データ取込ページ
# -----------------------------------------------------------------------------
if page == "データ取込":
    st.subheader("データプレビュー")
    st.dataframe(_df, use_container_width=True)
    st.write(f"📏 **{_df.shape[0]} 行 × {_df.shape[1]} 列**")

# -----------------------------------------------------------------------------
# 2. EDA ページ
# -----------------------------------------------------------------------------
if page == "EDA":
    st.subheader("基本統計量 & 欠損確認")

    st.write("### 基本統計量 (describe)")
    st.dataframe(_df.describe(include="all").transpose(), use_container_width=True)

    st.write("### 欠損値割合")
    na_pct = _df.isna().mean().sort_values(ascending=False) * 100
    na_df = na_pct.to_frame("Missing %")
    st.dataframe(na_df, use_container_width=True, height=300)

    # 欠損ヒートマップ
    st.write("### 欠損ヒートマップ")
    fig_na = px.imshow(_df.isna(), aspect="auto", color_continuous_scale=["#eeeeee", "#ff6961"])
    st.plotly_chart(fig_na, use_container_width=True)

    # 相関ヒートマップ
    numeric_cols = _df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.write("### 相関係数ヒートマップ (Pearson)")
        corr = _df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# 3. 統計解析ページ
# -----------------------------------------------------------------------------
if page == "統計解析":
    st.subheader("統計解析メニュー")
    analysis_type = st.selectbox("解析タイプを選択", [
        "因子分析 (Factor Analysis)",
        "線形回帰モデル"
    ])

    # 共通: 解析用数値列選択
    numeric_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("数値列がありません。別のデータセットをアップロードしてください。")
        st.stop()

    if analysis_type.startswith("因子分析"):
        st.write("#### 因子分析の前処理")
        selected = st.multiselect("使用する数値列 (2列以上)", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
        n_factors = st.slider("抽出する因子の数", 1, min(len(selected), 10), 2)
        if st.button("実行", key="fa_run"):
            try:
                fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
                fa.fit(_df[selected].dropna())
                loadings = pd.DataFrame(fa.loadings_, index=selected, columns=[f"Factor{i+1}" for i in range(n_factors)])
                st.write("### 因子負荷量")
                st.dataframe(loadings, use_container_width=True)

                # スクリープロット
                st.write("### 固有値 (Scree Plot)")
                ev, _ = fa.get_eigenvalues()
                fig_scree = px.line(x=np.arange(1, len(ev) + 1), y=ev, markers=True, labels={"x": "Factor", "y": "Eigenvalue"})
                st.plotly_chart(fig_scree, use_container_width=True)

                st.session_state.report_text = loadings.to_csv()
            except Exception as e:
                st.toast(f"因子分析エラー: {e}", icon="❌")

    elif analysis_type == "線形回帰モデル":
        st.write("#### 回帰設定")
        target = st.selectbox("目的変数 (y)", numeric_cols)
        features = st.multiselect("説明変数 (X)", [c for c in numeric_cols if c != target], default=[c for c in numeric_cols if c != target])
        test_size = st.slider("テストデータ割合", 0.1, 0.5, 0.2, step=0.05)
        if st.button("実行", key="lr_run"):
            try:
                X_train, X_test, y_train, y_test = train_test_split(_df[features].dropna(), _df[target].dropna(), test_size=test_size, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                st.success(f"R²: {score:.3f}")

                coeff_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
                st.dataframe(coeff_df, use_container_width=True)

                st.session_state.report_text = coeff_df.to_csv()
            except Exception as e:
                st.toast(f"回帰モデルエラー: {e}", icon="❌")

    # ダウンロードボタン
    if st.session_state.report_text:
        st.download_button(
            "⬇️ 解析結果を CSV ダウンロード",
            data=st.session_state.report_text,
            file_name="analysis_result.csv",
            mime="text/csv",
        )

# -----------------------------------------------------------------------------
# 4. 可視化ページ
# -----------------------------------------------------------------------------
if page == "可視化":
    st.subheader("インタラクティブ可視化")

    numeric_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("数値列が2つ以上必要です。")
        st.stop()

    x_axis = st.selectbox("X 軸", numeric_cols, index=0)
    y_axis = st.selectbox("Y 軸", [c for c in numeric_cols if c != x_axis], index=1)
    color_col = st.selectbox("カラー分類 (任意)", [None] + _df.columns.tolist())

    fig_scatter = px.scatter(_df, x=x_axis, y=y_axis, color=color_col, title="Scatter Plot")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.write("### ヒストグラム / KDE")
    hist_col = st.selectbox("ヒストグラム対象列", numeric_cols)
    fig_hist = px.histogram(_df, x=hist_col, marginal="box", nbins=30)
    st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------------------------------------------
# フッター
# -----------------------------------------------------------------------------
st.caption("Made with Streamlit · Cached with @st.cache_data · Theme customizable via .streamlit/config.toml")
