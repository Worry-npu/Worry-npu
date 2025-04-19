import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🔍 DBSCAN 密度聚类")
uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv"])
eps = st.sidebar.slider("邻域半径 eps", 0.1, 5.0, step=0.1, value=0.5)
min_samples = st.sidebar.slider("最小样本数", 1, 10, value=5)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) < 2:
        st.warning("需要至少两个数值特征")
    else:
        X_scaled = StandardScaler().fit_transform(df[numeric_cols])
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        df['Cluster'] = labels

        st.write(df.head())
        fig, ax = plt.subplots()
        sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette='tab10')
        plt.title("DBSCAN 聚类结果")
        st.pyplot(fig)
else:
    st.info("请上传数据集")