import streamlit as st

st.set_page_config(page_title="数据挖掘算法原型", layout="wide")

st.title("💡 欢迎使用数据挖掘原型系统")
st.markdown("""
在左侧菜单选择不同的功能模块，包括：

1_KMeans聚类
2_DBSCAN聚类
3_层次聚类
4_MeanShift
5_DecisionTree分类
6_SVM分类
7_RandomForest分类
8_KNN分类
9_Logistic回归
10_NaiveBayes
11_Linear回归
12_DecisionTree回归
13_RandomForest回归
14_XGBoost回归
15_PCA降维
16_tSNE降维
17_AutoEncoder降维
18_IsolationForest
19_Apriori关联
20_FPGrowth关联

上传数据并开始分析吧！
""")
