import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="XGBoost 回归分析", layout="wide")
st.title("📈 XGBoost 回归建模与分析")

# 数据上传
uploaded_file = st.sidebar.file_uploader("📂 上传数据文件 (CSV 格式)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 数据预览")
    st.dataframe(df.head(10))

    with st.sidebar:
        st.markdown("### 🔧 模型参数设置")
        target_col = st.selectbox("🎯 选择目标变量（Y）", df.columns)
        feature_cols = st.multiselect("📊 选择特征变量（X）", [col for col in df.columns if col != target_col])
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
        max_depth = st.slider("max_depth", 1, 10, 4)
        n_estimators = st.slider("n_estimators", 10, 500, 100, step=10)
        learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01)

    if feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 模型训练
        model = xgb.XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            random_state=42
        )
        model.fit(X_train, y_train)

        # 模型预测
        y_pred = model.predict(X_test)

        # 模型评估
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 模型评估结果")
        st.markdown(f"**RMSE（均方根误差）**: `{rmse:.4f}`")
        st.markdown(f"**R²（拟合优度）**: `{r2:.4f}`")

        # 可视化：预测 vs 实际
        st.subheader("📉 实际值 vs 预测值")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='dodgerblue', alpha=0.7)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        ax.set_xlabel("实际值")
        ax.set_ylabel("预测值")
        ax.set_title("预测结果对比")
        st.pyplot(fig)

        # 特征重要性
        st.subheader("⭐ 特征重要性")
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        st.dataframe(importance_df)

        fig2, ax2 = plt.subplots()
        ax2.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
        ax2.invert_yaxis()
        ax2.set_title("特征重要性图")
        st.pyplot(fig2)
    else:
        st.warning("⚠️ 请先选择特征变量。")
else:
    st.info("📥 请上传一个包含数值特征和目标列的CSV文件。")
