import pandas as pd
import streamlit as st
import plotly.express as px

from utils.pdf_utils import make_pdf, canvas

st.set_page_config(page_title="グラフ一覧", layout="wide")
st.title("グラフ一覧")
st.write(
    "CSVをアップロードし、可視化・統計解析・因子分析・重回帰分析を行えます.\n\n" 
    "💡 サイドバーから各分析ページへ移動できます。"
)

uploaded_file = st.sidebar.file_uploader("📂 CSVファイルをアップロード", type="csv")

df: pd.DataFrame | None = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding="shift_jis")
        st.session_state['df'] = df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
            st.session_state['df'] = df
        except Exception:
            st.toast("⚠️ このファイルは使用できません。CSV 形式を確認してください。", icon="⚠️")
            st.stop()
else:
    st.info("左のサイドバーからCSVファイルをアップロードしてください。")


if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("📋 データプレビュー")
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 各項目の可視化・統計")
    
    visualization_report_content = []
    visualization_report_content.append("【データ可視化レポート】")
    visualization_report_content.append("")
    visualization_report_content.append("データプレビュー:")
    visualization_report_content.append(df.head().to_string(index=False, header=True))
    visualization_report_content.append("")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        st.markdown(f"### 🔹 {col}")
        col1, col2 = st.columns([2, 1])
        
        visualization_report_content.append(f"🔹 {col}")
        visualization_report_content.append("")

        with col1:
            fig_line = px.line(df, y=col, title=f"{col} の推移", markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
            visualization_report_content.append(fig_line)

            fig_hist = px.histogram(df, x=col, title=f"{col} の度数分布")
            st.plotly_chart(fig_hist, use_container_width=True)
            visualization_report_content.append(fig_hist)

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
    visualization_report_content.append(fig_corr)
    visualization_report_content.append("")

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
