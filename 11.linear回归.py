import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 线性回归分析", layout="wide")
st.title("📈 Linear Regression 回归建模与分析")

# 上传数据
uploaded_file = st.sidebar.file_uploader("📂 上传CSV文件", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 数据预览")
    st.dataframe(df.head(10))

    with st.sidebar:
        st.markdown("### 🔧 模型参数设置")
        target_col = st.selectbox("🎯 选择目标列", df.columns)
        feature_cols = st.multiselect("📊 选择特征列", [col for col in df.columns if col != target_col])
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)

    if feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 建立模型
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 模型评估
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 模型评估")
        st.markdown(f"**RMSE（均方根误差）**: `{rmse:.4f}`")
        st.markdown(f"**R²（拟合优度）**: `{r2:.4f}`")

        # 实际 vs 预测图
        st.subheader("📉 实际值 vs 预测值")
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, y_pred, color='blue', alpha=0.7)
        ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        ax1.set_xlabel("实际值")
        ax1.set_ylabel("预测值")
        ax1.set_title("实际值 vs 预测值")
        st.pyplot(fig1)

        # 系数可视化
        st.subheader("📐 回归系数（特征权重）")
        coef_df = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': model.coef_
        }).sort_values(by='Coefficient', ascending=False)

        st.dataframe(coef_df)

        fig2, ax2 = plt.subplots()
        ax2.barh(coef_df['Feature'], coef_df['Coefficient'], color='skyblue')
        ax2.set_title("回归系数")
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.invert_yaxis()
        st.pyplot(fig2)
    else:
        st.warning("⚠️ 请先选择特征变量")
else:
    st.info("📥 请上传CSV格式的数据文件。")
