import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.title("🔗 Apriori 关联分析")

uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # 假设数据已经是事务数据格式，需要转为one-hot编码
    df_onehot = pd.get_dummies(df)

    # 进行频繁项集挖掘
    frequent_itemsets = apriori(df_onehot, min_support=0.1, use_colnames=True)

    # 计算关联规则
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    st.write("关联规则：")
    st.dataframe(rules)
else:
    st.info("请上传一个CSV文件。")
