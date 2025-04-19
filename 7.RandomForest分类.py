import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🌲 Random Forest 分类", layout="wide")
st.title("🌲 Random Forest 分类模型分析")

# 上传数据
uploaded_file = st.sidebar.file_uploader("📂 上传CSV文件", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 数据预览")
    st.dataframe(df.head(10))

    with st.sidebar:
        st.markdown("### ⚙️ 模型配置")
        target_col = st.selectbox("🎯 选择目标列", df.columns)
        feature_cols = st.multiselect("📊 选择特征列", [col for col in df.columns if col != target_col])
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
        n_estimators = st.slider("🌲 树的数量 (n_estimators)", 10, 200, 100, 10)
        max_depth = st.slider("📏 最大深度 (max_depth)", 1, 50, 10)

    if feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        # 编码字符串类型
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = LabelEncoder().fit_transform(X[col])
        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # 划分训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 模型训练
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 模型评估
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.subheader("✅ 模型评估结果")
        st.markdown(f"**准确率**：`{acc:.4f}`")
        st.dataframe(report_df.style.background_gradient(cmap="Greens"))

        st.subheader("📊 混淆矩阵")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
        ax.set_xlabel("预测值")
        ax.set_ylabel("实际值")
        ax.set_title("混淆矩阵")
        st.pyplot(fig)

    else:
        st.warning("⚠️ 请至少选择一个特征列")
else:
    st.info("📥 请上传CSV格式的数据文件。")
