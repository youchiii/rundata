import streamlit as st

# --- Authentication Check ---
if 'user' not in st.session_state or st.session_state['user']['role'] != 'student':
    st.error("このページにアクセスする権限がありません。")
    st.info("生徒アカウントでログインしてください。")
    st.stop()

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

st.title("🧠 因子分析")

if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("🧠 因子分析")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
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
else:
    st.info("メインページでCSVファイルをアップロードしてください。")