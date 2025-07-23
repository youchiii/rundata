import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# PDF 生成用
# ------------------------------------------------------------
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # 日本語フォントの登録
    # ここに日本語フォントファイル（.ttf）のパスを指定してください。
    # 例: 'ipaexg.ttf' がアプリと同じディレクトリにある場合
    FONT_PATH = 'ipaexg.ttf' # または '/path/to/your/font/ipaexg.ttf'
    FONT_NAME = 'IPAexGothic'

    try:
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
        FONT_REGISTERED = True
    except Exception as e:
        st.warning(f"日本語フォントの登録に失敗しました: {e}。PDFで文字化けする可能性があります。")
        FONT_REGISTERED = False

except ModuleNotFoundError:
    canvas = None
    FONT_REGISTERED = False # reportlab自体がない場合もフォントは登録されない


# ------------------------------------------------------------
# ページ設定
# ------------------------------------------------------------
st.set_page_config(page_title="時系列データ解析アプリ", layout="wide")
st.title("🏃‍♂️ マラソンデータ解析アプリ")
st.write(
    "CSVをアップロードし、可視化・統計解析・因子分析・重回帰分析を行えます。\n\n"
    "💡 **URL共有ボタン** で現在のページを URL パラメータに保存 → アドレスバーをコピーするとワンクリック共有ができます。"
)

# ------------------------------------------------------------
# アップロード & クエリパラメータ処理
# ------------------------------------------------------------
params = st.experimental_get_query_params()
page_param = params.get("page", [None])[0]

uploaded_file = st.sidebar.file_uploader("📂 CSVファイルをアップロード", type="csv")

df: pd.DataFrame | None = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding="shift_jis")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except Exception:
            st.toast("⚠️ このファイルは使用できません。CSV 形式を確認してください。", icon="⚠️")
            st.stop()
else:
    st.info("左のサイドバーからCSVファイルをアップロードしてください。")

# ------------------------------------------------------------
# ページ選択
# ------------------------------------------------------------
pages = ["📈 データ可視化", "📊 統計解析", "📉 重回帰分析", "🧠 因子分析"]
page_default = pages.index(page_param) if page_param in pages else 0
page = st.sidebar.radio("📄 表示ページを選択", pages, index=page_default)

# URL 共有ボタン
if st.sidebar.button("🔗 このビューを共有"):
    st.experimental_set_query_params(page=page)
    st.sidebar.success("URLを更新しました。アドレスバーからコピーして共有してください。")

# ------------------------------------------------------------
# PDF 出力ユーティリティ
# ------------------------------------------------------------

def make_pdf(content: str) -> io.BytesIO | None:
    """与えられた文字列 content を PDF にして返す。reportlab が無ければ None"""
    if canvas is None:
        return None
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    line_height = 14
    
    # 日本語フォントが登録されていれば設定
    if FONT_REGISTERED:
        c.setFont(FONT_NAME, 12) # フォントサイズも指定
        line_height = 14 # フォントサイズに合わせて行の高さを調整
    else:
        c.setFont('Helvetica', 12) # デフォルトフォント

    y = height - 40
    for line in content.splitlines():
        # 日本語フォントが登録されていない場合は、文字化けを避けるためASCII文字のみを考慮
        # または、より広い幅で折り返す
        wrap_width = int((width - 80) / (6 if FONT_REGISTERED else 8)) # 日本語フォントがなければ1文字あたりの幅を小さく見積もる
        wrapped_lines = textwrap.wrap(line, width=wrap_width)
        for w_line in wrapped_lines:
            if y < 40:
                c.showPage()
                y = height - 40
                if FONT_REGISTERED:
                    c.setFont(FONT_NAME, 12) # 改ページ後もフォントを再設定
            c.drawString(40, y, w_line)
            y -= line_height
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------------------------------------------
# メインロジック
# ------------------------------------------------------------
if df is not None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [col for col in df.columns if df[col].nunique() < 10]

    # ----------------------------------------------------- 可視化ページ
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
        st.plotly_chart(
            px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale="Blues"),
            use_container_width=True,
        )

    # ----------------------------------------------------- 統計解析ページ
    elif page == "📊 統計解析":
        report_lines: list[str] = []

        st.subheader("🔁 クロス集計・χ²検定")
        if len(cat_cols) >= 2:
            cat1 = st.selectbox("カテゴリ列①", cat_cols, key="cat1")
            cat2 = st.selectbox("カテゴリ列②", cat_cols, key="cat2")
            if st.button("χ²検定を実行"):
                ctab = pd.crosstab(df[cat1], df[cat2])
                st.dataframe(ctab)
                chi2, p, dof, _ = chi2_contingency(ctab)
                msg = f"χ²統計量: {chi2:.4f}, 自由度: {dof}, p値: {p:.4f}"
                st.write(msg)
                sig = "有意な差あり（p < 0.05）" if p < 0.05 else "有意な差なし"
                st.success(sig)
                report_lines.extend([
                    "【χ²検定】",
                    f"カテゴリ列1: {cat1}, カテゴリ列2: {cat2}",
                    msg,
                    sig,
                    "",
                    "クロス集計:",
                    ctab.to_string(index=True, header=True),
                    "",
                    "",
                ])
        else:
            st.info("χ²検定にはカテゴリ列が2つ以上必要です。")


        st.subheader("📐 t検定（2群）")
        if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
            num_col = st.selectbox("数値列", numeric_cols)
            group_col = st.selectbox("グループ列", cat_cols)
            groups = df[group_col].dropna().unique()
            if len(groups) == 2:
                if st.button("t検定を実行", key="t_test_button"):
                    g1, g2 = groups
                    data1 = df[df[group_col] == g1][num_col].dropna()
                    data2 = df[df[group_col] == g2][num_col].dropna()

                    if not data1.empty and not data2.empty:
                        t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
                        msg = f"{g1} vs {g2}  |  t統計量: {t_stat:.4f}, p値: {p_val:.4f}"
                        st.write(msg)
                        sig = "有意差あり（p < 0.05）" if p_val < 0.05 else "有意差なし"
                        st.success(sig)
                        report_lines.extend([
                            "【t検定】",
                            f"数値列: {num_col}, グループ列: {group_col}",
                            msg,
                            sig,
                            "",
                            "",
                        ])
                    else:
                        st.warning("選択されたグループに有効なデータがありません。")
            elif len(groups) != 2:
                st.info("t検定を行うには、グループ列に2つのユニークな値が必要です。")
        else:
            st.info("t検定にはカテゴリ列が1つ以上、数値列が1つ以上必要です。")


        # ---------- PDF レポートのダウンロードボタン
        if canvas and report_lines:
            pdf_buf = make_pdf("\n".join(report_lines))
            st.download_button(
                "📄 PDFレポートをダウンロード",
                pdf_buf,
                file_name="stats_report.pdf",
                mime="application/pdf"
            )
        elif not canvas:
            st.info("PDF 機能には `reportlab` ライブラリを追加してください。`pip install reportlab` でインストールできます。")
        else:
            st.info("PDFレポートを生成するには、統計解析を実行してください。")


    # ----------------------------------------------------- 重回帰分析ページ
    elif page == "📉 重回帰分析":
        regression_report_lines: list[str] = []

        if len(numeric_cols) >= 2:
            st.subheader("📉 重回帰分析")
            target = st.selectbox("🎯 目的変数", numeric_cols)
            features = st.multiselect("🧮 説明変数", [col for col in numeric_cols if col != target])
            if features:
                X = df[features].dropna()
                y = df[target].loc[X.index]

                if not X.empty and not y.empty:
                    model = LinearRegression()

                    if st.button("重回帰分析を実行"):
                        if len(X) < 2:
                            st.warning("データ数が少なすぎるため、重回帰分析を実行できません。")
                        else:
                            test_size = 0.2 if len(X) * 0.2 >= 1 else 0.5
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)

                            st.markdown("#### ✅ 結果")
                            st.write(f"決定係数（R²）: {r2:.4f}")
                            st.write(f"平均二乗誤差（MSE）: {mse:.4f}")

                            coef_df = pd.DataFrame({"変数": features, "係数": model.coef_})
                            st.dataframe(coef_df)

                            equation = " + ".join([f"{coef:.2f}×{var}" for coef, var in zip(model.coef_, features)])
                            full_equation = f"{target} = {model.intercept_:.2f} + {equation}"
                            st.markdown(f"#### 📏 回帰式：{full_equation}")

                            st.markdown("#### 実測 vs 予測")
                            fig = px.scatter(
                                x=y_test,
                                y=y_pred,
                                labels={'x': '実測値', 'y': '予測値'},
                                title='実測値 vs 予測値',
                                trendline="ols",
                                trendline_color_override="red"
                            )
                            min_val = min(y_test.min(), y_pred.min())
                            max_val = max(y_test.max(), y_pred.max())
                            fig.add_shape(
                                type="line", line=dict(dash="dash"),
                                x0=min_val, y0=min_val, x1=max_val, y1=max_val
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            st.markdown("---")
                            st.markdown("#### 予測結果の詳細")
                            results_df = pd.DataFrame({'実測値': y_test, '予測値': y_pred})
                            st.dataframe(results_df)

                            regression_report_lines.extend([
                                "【重回帰分析レポート】",
                                f"目的変数: {target}",
                                f"説明変数: {', '.join(features)}",
                                "",
                                f"決定係数（R²）: {r2:.4f}",
                                f"平均二乗誤差（MSE）: {mse:.4f}",
                                "",
                                f"回帰式: {full_equation}",
                                "",
                                "係数:",
                                coef_df.to_string(index=False, header=True),
                                "",
                                "",
                            ])

                            if canvas and regression_report_lines:
                                pdf_buf = make_pdf("\n".join(regression_report_lines))
                                st.download_button(
                                    "📄 重回帰分析レポートをダウンロード",
                                    pdf_buf,
                                    file_name="regression_report.pdf",
                                    mime="application/pdf"
                                )
                            elif not canvas:
                                st.info("PDF 機能には `reportlab` ライブラリを追加してください。`pip install reportlab` でインストールできます。")
                            else:
                                st.info("PDFレポートを生成するには、重回帰分析を実行してください。")

                else:
                    st.info("重回帰分析を実行するには、説明変数を選択し、「重回帰分析を実行」ボタンをクリックしてください。")
            else:
                st.info("説明変数を選択してください。")
        else:
            st.info("重回帰分析には数値列が2つ以上必要です。")


    # ----------------------------------------------------- 因子分析ページ
    elif page == "🧠 因子分析":
        st.subheader("🧠 因子分析")
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("因子分析を行う変数を選択", numeric_cols, default=numeric_cols)
            if selected_cols:
                max_components = len(selected_cols) - 1
                if max_components < 1:
                    st.warning("因子分析を行うには、選択された変数が2つ以上必要です。")
                else:
                    n_components = st.slider("因子の数", min_value=1, max_value=max_components, value=min(2, max_components))
                    if st.button("因子分析を実行"):
                        data_for_fa = df[selected_cols].dropna()
                        if not data_for_fa.empty:
                            if len(data_for_fa) < 2:
                                st.warning("因子分析を行うには、有効なデータが2行以上必要です。")
                            else:
                                fa = FactorAnalysis(n_components=n_components, random_state=42)
                                fa.fit(data_for_fa)

                                st.markdown("#### 因子負荷量")
                                factor_loadings = pd.DataFrame(fa.components_.T, index=selected_cols, columns=[f"因子{i+1}" for i in range(n_components)])
                                st.dataframe(factor_loadings)

                                st.markdown("#### 因子得点 (上位5件)")
                                factor_scores = pd.DataFrame(fa.transform(data_for_fa), columns=[f"因子{i+1}" for i in range(n_components)], index=data_for_fa.index)
                                st.dataframe(factor_scores.head())

                                st.markdown("#### 寄与率")
                                eigenvalues = np.sum(fa.components_**2, axis=1)
                                total_variance = np.sum(eigenvalues)
                                if total_variance > 0:
                                    explained_variance_ratio = eigenvalues / total_variance
                                    st.write(f"各因子の寄与率: {explained_variance_ratio}")
                                    st.write(f"累積寄与率: {np.cumsum(explained_variance_ratio)}")
                                else:
                                    st.write("寄与率を計算できませんでした。")
                        else:
                            st.warning("選択された変数に有効なデータがありません。")
            else:
                st.info("因子分析を行う変数を選択してください。")
        else:
            st.info("因子分析には数値列が2つ以上必要です。")
