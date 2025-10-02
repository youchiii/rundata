import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils.pdf_utils import make_pdf, canvas

st.title("📉 重回帰分析")

if 'df' in st.session_state:
    df = st.session_state['df']
    regression_report_lines: list = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

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
                        fig_scatter = px.scatter(
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
                            fig_scatter,
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
else:
    st.info("メインページでCSVファイルをアップロードしてください。")