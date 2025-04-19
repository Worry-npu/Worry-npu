import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="FP-Growth关联规则分析", layout="wide")

st.title("📊 FP-Growth 关联规则分析系统")

# 上传CSV数据
st.sidebar.header("📂 数据上传与参数设置")
uploaded_file = st.sidebar.file_uploader("上传交易数据 (CSV 格式，每行一个交易项集)", type=["csv"])

min_support = st.sidebar.slider("最小支持度 (support)", 0.1, 1.0, 0.3, 0.05)
min_confidence = st.sidebar.slider("最小置信度 (confidence)", 0.1, 1.0, 0.7, 0.05)

def preprocess_data(df):
    """ 将交易数据转为 one-hot 编码格式 """
    transactions = df.stack().groupby(level=0).apply(list)
    all_items = sorted(set(item for sublist in transactions for item in sublist))
    encoded_rows = [{item: (item in row) for item in all_items} for row in transactions]
    return pd.DataFrame(encoded_rows)

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file, header=None)
    st.subheader("📌 原始交易数据")
    st.dataframe(raw_df.head(10))

    onehot_df = preprocess_data(raw_df)

    # FP-Growth 分析
    st.subheader("📈 FP-Growth 分析结果")
    freq_itemsets = fpgrowth(onehot_df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("✅ **频繁项集**")
        st.dataframe(freq_itemsets.sort_values(by="support", ascending=False))

    with col2:
        st.markdown("📐 **关联规则**")
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

    # 可视化：频繁项集柱状图
    st.subheader("📊 频繁项集支持度分布图")
    fig, ax = plt.subplots()
    top_items = freq_itemsets.copy()
    top_items["itemsets"] = top_items["itemsets"].apply(lambda x: ', '.join(x))
    top_items.sort_values(by="support", ascending=False).head(10).plot.bar(
        x="itemsets", y="support", ax=ax, legend=False, color="skyblue"
    )
    ax.set_ylabel("支持度")
    ax.set_xlabel("频繁项集")
    ax.set_title("Top 10 频繁项集支持度")
    st.pyplot(fig)

    # 可视化：网络图
    st.subheader("🌐 关联规则网络图")
    G = nx.DiGraph()
    for _, row in rules.iterrows():
        antecedent = ', '.join(row["antecedents"])
        consequent = ', '.join(row["consequents"])
        G.add_node(antecedent)
        G.add_node(consequent)
        G.add_edge(antecedent, consequent, weight=row["confidence"])

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G, k=0.8, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, ax=ax2)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax2)
    st.pyplot(fig2)

else:
    st.info("请上传CSV交易数据文件，每行一个交易项（多列为多个商品项）。示例：牛奶,面包,尿布")

