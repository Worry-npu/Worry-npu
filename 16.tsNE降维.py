import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("🔍 t-SNE 降维")

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

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)

        st.write("t-SNE结果：")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1])
        ax.set_xlabel("t-SNE 维度1")
        ax.set_ylabel("t-SNE 维度2")
        ax.set_title("t-SNE 降维结果")
        st.pyplot(fig)
else:
    st.info("请上传一个CSV文件。")
