import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.title("🔍 Isolation Forest 异常检测")

uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("请上传包含至少两个数值特征的CSV文件")
    else:
        clf = IsolationForest(contamination=0.1)
        df_outliers = clf.fit_predict(df[numeric_cols])

        # 标记异常点
        df['outlier'] = df_outliers
        st.write(f"检测到 {df_outlier.sum()} 个异常点。")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df.index, df[numeric_cols[0]], c=df['outlier'], cmap='coolwarm')
        ax.set_xlabel("样本索引")
        ax.set_ylabel(numeric_cols[0])
        ax.set_title("Isolation Forest 异常检测")
        st.pyplot(fig)
else:
    st.info("请上传一个CSV文件。")
