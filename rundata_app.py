import streamlit as st
import pandas as pd
import plotly.express as px

from auth.database import init_db
from utils.pdf_utils import make_pdf, canvas

# --- Initialize Database ---
# This will create the DB and the initial accounts on first run.
init_db()

# --- Page Configuration ---
st.set_page_config(page_title="データ解析アプリ", layout="wide")

# --- Authentication Router ---
if 'user' not in st.session_state:
    # If user is not logged in, redirect to login page
    st.switch_page("pages/1_Login.py")

# --- LOGGED IN --- 
user_info = st.session_state['user']

# Display user info in sidebar based on role
if user_info['role'] == 'teacher':
    st.sidebar.info("管理者でログイン中")
else:
    st.sidebar.info(f"{user_info['username']}でログイン中")

if st.sidebar.button("ログアウト", type="primary"):
    del st.session_state['user']
    st.switch_page("pages/1_Login.py")

# --- Role-based Page Access & Redirection ---
if user_info['role'] == 'teacher':
    # If a teacher lands here, redirect them to the Admin page
    st.switch_page("pages/Admin.py")

# --- Main app for students ---
# The rest of the script will only run if the user is a student.
st.title("🏃‍♂️ データ解析アプリ")
st.write("CSVをアップロードし、可視化・統計解析・因子分析・重回帰分析を行えます。")
st.write("💡 サイドバーから各分析ページへ移動できます。")

uploaded_file = st.sidebar.file_uploader("📂 CSVファイルをアップロード", type="csv")

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

if 'df' not in st.session_state:
    st.info("左のサイドバーからCSVファイルをアップロードしてください。")
    st.stop()

# --- If df exists, show the rest of the dashboard ---
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
    
    with col1:
        fig_line = px.line(df, y=col, title=f"{col} の推移", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

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

st.subheader("📊 相関係数マトリクス")
fig_corr = px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale="Blues", title="相関係数マトリクス")
st.plotly_chart(fig_corr, use_container_width=True)
