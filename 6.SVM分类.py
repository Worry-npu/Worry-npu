import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

st.title("📈 支持向量机 SVM 分类")
uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    all_cols = df.columns.tolist()
    label_col = st.sidebar.selectbox("选择目标变量 (Label)", all_cols)
    feature_cols = st.sidebar.multiselect("选择特征列", [col for col in all_cols if col != label_col])

    if len(feature_cols) > 0:
        X = df[feature_cols]
        y = df[label_col]

        # 独热编码与标准化
        X = pd.get_dummies(X)
        X = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = SVC(kernel='rbf')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.text("分类报告：")
        st.text(classification_report(y_test, y_pred))
else:
    st.info("请上传数据集")
