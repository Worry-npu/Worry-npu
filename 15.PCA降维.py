import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("📉 PCA 主成分分析")

uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("请上传包含至少两个数值特征的CSV文件")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols])
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)

        st.write("主成分解释方差比：", pca.explained_variance_ratio_)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(components[:, 0], components[:, 1])
        ax.set_xlabel("主成分1")
        ax.set_ylabel("主成分2")
        ax.set_title("PCA 降维结果")
        st.pyplot(fig)
else:
    st.info("请上传一个CSV文件。")
