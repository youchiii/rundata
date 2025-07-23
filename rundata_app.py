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
# ページ設定 (Streamlitコマンドの最上位に配置)
# ------------------------------------------------------------
st.set_page_config(page_title="時系列データ解析アプリ", layout="wide")
st.title("🏃‍♂️ マラソンデータ解析アプリ")
st.write(
    "CSVをアップロードし、可視化・統計解析・因子分析・重回帰分析を行えます。\n\n"
    "💡 **URL共有ボタン** で現在のページを URL パラメータに保存 → アドレスバーをコピーするとワンクリック共有ができます。"
)

# ------------------------------------------------------------
# PDF 生成用 (st.set_page_config() の後に移動)
# ------------------------------------------------------------
canvas = None
FONT_REGISTERED = False
FONT_NAME = 'IPAexGothic' # デフォルト名を設定

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rl_canvas # 名前衝突を避けるため別名でインポート
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader # 画像埋め込み用に追加
    
    canvas = rl_canvas # グローバル変数canvasにreportlabのCanvasクラスを割り当て

    # 日本語フォントの登録
    # ここに日本語フォントファイル（.ttf）のパスを指定してください。
    # 例: 'ipaexg.ttf' がアプリと同じディレクトリにある場合
    FONT_PATH = 'ipaexg.ttf' # または '/path/to/your/font/ipaexg.ttf'
    
    try:
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
        FONT_REGISTERED = True
    except Exception as e:
        st.warning(f"日本語フォントの登録に失敗しました: {e}。PDFで文字化けする可能性があります。")
        FONT_REGISTERED = False

except ModuleNotFoundError:
    st.info("PDF 機能には `reportlab` ライブラリを追加してください。`pip install reportlab` でインストールできます。")
    canvas = None
    FONT_REGISTERED = False


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

# make_pdf関数を修正し、テキストとPlotly Figureの両方を受け取れるようにする
def make_pdf(content_list: list) -> io.BytesIO | None:
    """与えられたコンテンツリスト（文字列またはPlotly Figure）をPDFにして返す。"""
    if canvas is None:
        return None
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    line_height = 14
    
    # 日本語フォントが登録されていれば設定
    if FONT_REGISTERED:
        c.setFont(FONT_NAME, 12)
        line_height = 14
    else:
        c.setFont('Helvetica', 12)

    y = height - 40 # 初期Y座標

    for item in content_list:
        if isinstance(item, str):
            # テキストの場合
            wrap_width = int((width - 80) / (6 if FONT_REGISTERED else 8))
            wrapped_lines = textwrap.wrap(item, width=wrap_width)
            for w_line in wrapped_lines:
                if y < 40: # 改ページ
                    c.showPage()
                    y = height - 40
                    if FONT_REGISTERED:
                        c.setFont(FONT_NAME, 12) # 改ページ後もフォントを再設定
                c.drawString(40, y, w_line)
                y -= line_height
            y -= line_height # テキストブロックの後に少しスペースを開ける
        elif hasattr(item, 'write_image'): # Plotly Figureオブジェクトの場合
            # グラフをPNG画像としてBytesIOに書き出す
            img_buffer = io.BytesIO()
            try:
                # グラフの幅と高さを調整（PDFページに収まるように）
                # ここではA4横幅-余白に合わせるため、約500pxに設定
                img_width = 500
                img_height = int(item.height * (img_width / item.width)) if item.width else 300
                
                item.write_image(img_buffer, format='png', width=img_width, height=img_height, scale=1)
                img_buffer.seek(0)
                img = ImageReader(img_buffer)

                # 画像が現在のページに収まるか確認
                if y - img_height < 40: # 画像がページの下端を超える場合
                    c.showPage()
                    y = height - 40 # 新しいページの先頭に移動
                    if FONT_REGISTERED:
                        c.setFont(FONT_NAME, 12) # 改ページ後もフォントを再設定
                
                # 画像をPDFに描画
                # X座標は中央揃えにするため (width - img_width) / 2
                c.drawImage(img, (width - img_width) / 2, y - img_height, width=img_width, height=img_height)
                y -= (img_height + line_height) # 画像の高さ分と少しスペースをY座標から引く
            except Exception as e:
                st.warning(f"グラフのPDF埋め込みに失敗しました: {e}。`kaleido`がインストールされているか確認してください。")
                if y < 40:
                    c.showPage()
                    y = height - 40
                    if FONT_REGISTERED:
                        c.setFont(FONT_NAME, 12)
                c.drawString(40, y, f"[グラフの埋め込みに失敗しました: {item.layout.title.text if item.layout.title else '無題のグラフ'}]")
                y -= line_height
        else:
            # 未知のタイプの場合、文字列として扱う
            if y < 40:
                c.showPage()
                y = height - 40
                if FONT_REGISTERED:
                    c.setFont(FONT_NAME, 12)
            c.drawString(40, y, str(item))
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

        st.subheader("� 各項目の可視化・統計")
        
        # PDFレポートの内容を収集するリスト
        visualization_report_content = []
        visualization_report_content.append("【データ可視化レポート】")
        visualization_report_content.append("")
        visualization_report_content.append("データプレビュー:")
        visualization_report_content.append(df.head().to_string(index=False, header=True)) # 先頭5行のみ
        visualization_report_content.append("")

        for col in numeric_cols:
            st.markdown(f"### 🔹 {col}")
            col1, col2 = st.columns([2, 1])
            
            # 各項目の可視化・統計のPDF内容を収集
            visualization_report_content.append(f"🔹 {col}")
            visualization_report_content.append("")

            with col1:
                # 時系列推移グラフ
                fig_line = px.line(df, y=col, title=f"{col} の推移", markers=True)
                st.plotly_chart(fig_line, use_container_width=True)
                visualization_report_content.append(fig_line) # 図オブジェクトをリストに追加

                # 度数分布ヒストグラム
                fig_hist = px.histogram(df, x=col, title=f"{col} の度数分布")
                st.plotly_chart(fig_hist, use_container_width=True)
                visualization_report_content.append(fig_hist) # 図オブジェクトをリストに追加

            with col2:
                st.markdown("**基本統計量**")
                mean_val = df[col].mean()
                median_val = df[col].median()
                mode_val = df[col].mode().iat[0] if not df[col].mode().empty else "なし"
                var_val = df[col].var()
                std_val = df[col].std()

                st.write(f"- 平均: {mean_val:.2f}")
                st.write(f"- 中央値: {median_val:.2f}")
                st.write(f"- 最頻値: {mode_val}")
                st.write(f"- 分散: {var_val:.2f}")
                st.write(f"- 標準偏差: {std_val:.2f}")

                # 基本統計量をPDF内容に追加
                visualization_report_content.append(f"基本統計量 ({col}):")
                visualization_report_content.append(f"  - 平均: {mean_val:.2f}")
                visualization_report_content.append(f"  - 中央値: {median_val:.2f}")
                visualization_report_content.append(f"  - 最頻値: {mode_val}")
                visualization_report_content.append(f"  - 分散: {var_val:.2f}")
                visualization_report_content.append(f"  - 標準偏差: {std_val:.2f}")
                visualization_report_content.append("")

        st.subheader("📊 相関係数マトリクス")
        fig_corr = px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale="Blues", title="相関係数マトリクス")
        st.plotly_chart(fig_corr, use_container_width=True)
        visualization_report_content.append(fig_corr) # 図オブジェクトをリストに追加
        visualization_report_content.append("")

        # PDFレポートのダウンロードボタン (データ可視化ページ用)
        if canvas and visualization_report_content:
            pdf_buf = make_pdf(visualization_report_content)
            st.download_button(
                "📄 データ可視化レポートをダウンロード",
                pdf_buf,
                file_name="visualization_report.pdf",
                mime="application/pdf"
            )
        elif not canvas:
            st.info("PDF 機能には `reportlab` ライブラリと `kaleido` ライブラリを追加してください。`pip install reportlab kaleido` でインストールできます。")
        else:
            st.info("PDFレポートを生成するには、データをアップロードしてください。")


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
            pdf_buf = make_pdf(report_lines) # make_pdfはリストを受け取る
            st.download_button(
                "📄 PDFレポートをダウンロード",
                pdf_buf,
                file_name="stats_report.pdf",
                mime="application/pdf"
            )
        elif not canvas:
            st.info("PDF 機能には `reportlab` ライブラリと `kaleido` ライブラリを追加してください。`pip install reportlab kaleido` でインストールできます。")
        else:
            st.info("PDFレポートを生成するには、統計解析を実行してください。")


    # ----------------------------------------------------- 重回帰分析ページ
    elif page == "📉 重回帰分析":
        regression_report_lines: list = [] # strだけでなくFigureも入る可能性があるので型ヒントをlistに

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
                            fig_scatter = px.scatter( # 変数名をfig_scatterに変更
                                x=y_test,
                                y=y_pred,
                                labels={'x': '実測値', 'y': '予測値'},
                                title='実測値 vs 予測値',
                                trendline="ols",
                                trendline_color_override="red"
                            )
                            min_val = min(y_test.min(), y_pred.min())
                            max_val = max(y_test.max(), y_pred.max())
                            fig_scatter.add_shape(
                                type="line", line=dict(dash="dash"),
                                x0=min_val, y0=min_val, x1=max_val, y1=max_val
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)

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
                                "実測 vs 予測グラフ:",
                                fig_scatter, # 図オブジェクトをリストに追加
                                "",
                            ])

                            if canvas and regression_report_lines:
                                pdf_buf = make_pdf(regression_report_lines)
                                st.download_button(
                                    "📄 重回帰分析レポートをダウンロード",
                                    pdf_buf,
                                    file_name="regression_report.pdf",
                                    mime="application/pdf"
                                )
                            elif not canvas:
                                st.info("PDF 機能には `reportlab` ライブラリと `kaleido` ライブラリを追加してください。`pip install reportlab kaleido` でインストールできます。")
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