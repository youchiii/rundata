import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency, ttest_ind

from utils.pdf_utils import make_pdf, canvas

st.title("📊 統計解析")

if 'df' in st.session_state:
    df = st.session_state['df']
    report_lines: list[str] = []

    st.subheader("🔁 クロス集計・χ²検定")
    cat_cols = [col for col in df.columns if df[col].nunique() < 10]
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
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
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

    if canvas and report_lines:
        pdf_buf = make_pdf(report_lines)
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
else:
    st.info("メインページでCSVファイルをアップロードしてください。")