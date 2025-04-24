# hierarchical_clustering.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

METHOD_MAP = {
    'ward': '离差平方和法',
    'complete': '全连接法',
    'average': '平均连接法',
    'single': '单连接法'
}

DISTANCE_MAP = {
    'euclidean': '欧氏距离',
    'cityblock': '曼哈顿距离',
    'cosine': '余弦相似度'
}


def main():
    st.set_page_config(page_title="层次聚类系统", layout="wide")
    st.title('🌳 层次聚类分析系统')

    # ==================== 文件上传 ====================
    with st.expander("📁 数据上传", expanded=True):
        uploaded_file = st.file_uploader("上传CSV文件", type="csv")

    if uploaded_file:
        try:
            # ==================== 数据处理 ====================
            data = pd.read_csv(uploaded_file)
            numeric_cols = data.select_dtypes(include='number').columns.tolist()

            if len(numeric_cols) < 2:
                st.error("需至少2个数值列")
                return

            # ==================== 界面布局 ====================
            col1, col2 = st.columns([0.7, 0.3], gap="large")

            with col2:
                # ==================== 参数设置 ====================
                with st.expander("⚙️ 算法参数", expanded=True):
                    method = st.selectbox(
                        "链接方法",
                        options=list(METHOD_MAP.keys()),
                        format_func=lambda x: METHOD_MAP[x],
                        help="不同样本间距离计算方式"
                    )

                    distance = st.selectbox(
                        "距离度量",
                        options=['euclidean', 'cityblock', 'cosine'] if method != 'ward' else ['euclidean'],
                        format_func=lambda x: DISTANCE_MAP[x],
                        disabled=(method == 'ward'),
                        help="ward方法固定使用欧氏距离"
                    )

                    n_clusters = st.slider(
                        "聚类数量",
                        2, 10, 3,
                        help="根据树状图切割线位置调整"
                    )

            # ==================== 算法执行 ====================
            features = data[numeric_cols].dropna()
            scaled_data = StandardScaler().fit_transform(features)

            Z = linkage(scaled_data, method=method, metric=distance)
            labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1

            # ==================== 可视化 ====================
            with col1:
                tab1, tab2 = st.tabs(["🌿 树状图", "📊 特征空间"])

                with tab1:
                    fig1, ax = plt.subplots(figsize=(12, 5))
                    dendrogram(Z, ax=ax, orientation='top')
                    plt.axhline(y=plt.yticks()[0][-n_clusters + 1], color='r', linestyle='--')
                    plt.title(f"层次聚类树状图（{METHOD_MAP[method]}）", pad=15)
                    st.pyplot(fig1)

                with tab2:
                    if len(numeric_cols) >= 2:
                        fig2, ax = plt.subplots(figsize=(8, 6))
                        scatter = ax.scatter(
                            features.iloc[:, 0], features.iloc[:, 1],
                            c=labels, cmap='tab20', s=50, alpha=0.8
                        )
                        plt.colorbar(scatter).set_label('类别', rotation=270)
                        ax.set_xlabel(numeric_cols[0])
                        ax.set_ylabel(numeric_cols[1])
                        plt.grid(linestyle=':')
                        st.pyplot(fig2)

            # ==================== 分析报告 ====================
            with col2:
                with st.expander("📊 分析指标", expanded=True):
                    st.metric("轮廓系数", f"{silhouette_score(scaled_data, labels):.2f}")
                    st.metric("聚类中心数", n_clusters)

                    cluster_dist = pd.Series(labels).value_counts().to_dict()
                    st.write("**类别分布**")
                    st.json(cluster_dist)

                    st.download_button(
                        "下载报告",
                        generate_report(features, labels, method, distance),
                        file_name="hierarchical_report.md"
                    )

        except Exception as e:
            st.error(f"错误: {str(e)}")
            st.code("""
            常见问题排查：
            1. 检查缺失值：df.isnull().sum()
            2. 尝试不同距离度量
            3. 调整聚类数量""")


def generate_report(data, labels, method, distance):
    return f"""
    ## 层次聚类分析报告
    ### 参数配置
    - 链接方法：{METHOD_MAP[method]}
    - 距离度量：{DISTANCE_MAP[distance]}
    - 聚类数量：{len(np.unique(labels))}

    ### 数据统计
    - 样本数量：{len(data)}
    - 特征维度：{data.shape[1]}
    - 轮廓系数：{silhouette_score(data, labels):.2f}

    ### 类别分布
    {pd.Series(labels).value_counts().to_markdown()}
    """.encode('utf-8')


if __name__ == "__main__":
    main()
