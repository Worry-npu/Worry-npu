import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="📘 KNN 分类模型", layout="wide")
st.title("📘 KNN（K近邻）分类模型分析")

# 上传数据
uploaded_file = st.sidebar.file_uploader("📂 上传CSV文件", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 数据预览")
    st.dataframe(df.head(10))

    with st.sidebar:
        st.markdown("### ⚙️ 模型配置")
        target_col = st.selectbox("🎯 选择目标列（分类）", df.columns)
        feature_cols = st.multiselect("📊 选择特征列", [col for col in df.columns if col != target_col])
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
        k_neighbors = st.slider("K值（邻居数量）", 1, 20, 5)

    if feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        # 编码文本列
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = LabelEncoder().fit_transform(X[col])
        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 创建并训练模型
        model = KNeighborsClassifier(n_neighbors=k_neighbors)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 模型评估
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.subheader("📊 模型评估结果")
        st.markdown(f"**✅ 准确率**：`{acc:.4f}`")
        st.markdown("📋 分类报告：")
        st.dataframe(report_df.style.background_gradient(cmap="BuGn"))

        st.subheader("🧊 混淆矩阵")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("预测值")
        ax.set_ylabel("实际值")
        ax.set_title("混淆矩阵热力图")
        st.pyplot(fig)

    else:
        st.warning("⚠️ 请先选择特征变量")
else:
    st.info("📥 请上传CSV格式的数据文件。")
