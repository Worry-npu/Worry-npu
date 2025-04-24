# mean_shift.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import chardet
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


def safe_read_csv(uploaded_file):
    """安全读取CSV文件的多重验证方法"""
    try:
        # 第一次尝试自动检测编码
        raw_data = uploaded_file.getvalue()
        enc = chardet.detect(raw_data)['encoding']
        uploaded_file.seek(0)

        # 尝试常见中文编码
        encodings = [enc, 'utf-8', 'gbk', 'gb18030', 'latin1']

        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                if not df.empty:
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        # 如果所有编码都失败，尝试无header读取
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, header=None)

    except Exception as e:
        st.error(f"文件读取失败: {str(e)}")
        return pd.DataFrame()


def main():
    st.set_page_config(page_title="均值漂移系统", layout="wide")
    st.title('🌌 均值漂移聚类系统')

    # ==================== 文件上传增强模块 ====================
    with st.expander("📁 数据上传（支持CSV/Excel）", expanded=True):
        uploaded_file = st.file_uploader(
            "上传数据文件",
            type=['csv', 'xlsx'],
            help="建议使用UCI数据集，如客户消费数据"
        )

    if uploaded_file:
        # ==================== 数据加载验证 ====================
        if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            df = pd.read_excel(uploaded_file)
        else:
            df = safe_read_csv(uploaded_file)

        # 空数据校验
        if df.empty or len(df.columns) < 2:
            st.error("错误：文件无有效数据或列不足")
            return

        # 自动识别数值列
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("需要至少2个数值列进行分析")
            return

        # ==================== 界面布局 ====================
        col1, col2 = st.columns([0.7, 0.3], gap="large")

        # ==================== 参数设置模块 ====================
        with col2:
            with st.expander("⚙️ 算法参数配置", expanded=True):
                # 智能带宽估算
                bandwidth_type = st.radio(
                    "带宽模式",
                    options=['auto', 'manual'],
                    format_func=lambda x: "自动估算" if x == 'auto' else "手动设置",
                    help="自动模式根据数据分布智能计算"
                )

                if bandwidth_type == 'auto':
                    quantile = st.slider(
                        "数据分位数",
                        0.05, 0.5, 0.2,
                        help="分位值越小，检测的聚类越精细"
                    )
                else:
                    bandwidth = st.slider(
                        "带宽值",
                        0.1, 5.0, 2.0,
                        help="值越大，生成的聚类越少"
                    )

                # 动态参数建议
                if st.checkbox("显示参数建议"):
                    data_sample = df[numeric_cols].sample(100)
                    suggest_bandwidth = estimate_bandwidth(data_sample, quantile=0.2)
                    st.write(f"推荐带宽范围：{suggest_bandwidth * 0.5:.2f} ~ {suggest_bandwidth * 1.5:.2f}")

        # ==================== 算法核心模块 ====================
        try:
            # 数据预处理
            features = df[numeric_cols].dropna()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(features)

            # 带宽计算
            if bandwidth_type == 'auto':
                bandwidth = estimate_bandwidth(scaled_data, quantile=quantile)

            # 执行聚类
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            labels = ms.fit_predict(scaled_data)
            n_clusters = len(np.unique(labels))

            # ==================== 可视化模块 ====================
            with col1:
                tab1, tab2 = st.tabs(["📈 聚类分布", "📊 特征分析"])

                with tab1:
                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111)
                    scatter = ax.scatter(
                        features.iloc[:, 0],
                        features.iloc[:, 1],
                        c=labels,
                        cmap='tab20',
                        s=40,
                        alpha=0.7,
                        edgecolor='k'
                    )
                    plt.colorbar(scatter).set_label('聚类标签', rotation=270)
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel(numeric_cols[1])
                    ax.set_title(f"均值漂移聚类结果（带宽={bandwidth:.2f}）")
                    plt.grid(linestyle=':', alpha=0.6)
                    st.pyplot(fig)

                with tab2:
                    if len(numeric_cols) > 1:
                        fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
                        # 特征分布图
                        axes[0].scatter(features.iloc[:, 0], features.iloc[:, 1], alpha=0.3)
                        axes[0].set_title("原始特征分布")
                        # 聚类分布图
                        axes[1].scatter(features.iloc[:, 0], features.iloc[:, 1], c=labels, cmap='tab20')
                        axes[1].set_title("聚类结果分布")
                        st.pyplot(fig2)

            # ==================== 分析报告模块 ====================
            with col2:
                with st.expander("📝 分析报告", expanded=True):
                    st.metric("检测到聚类数", n_clusters)
                    st.metric("轮廓系数", f"{silhouette_score(scaled_data, labels):.2f}")

                    # 聚类中心展示
                    centers = scaler.inverse_transform(ms.cluster_centers_)
                    st.write("**聚类中心特征值**")
                    st.dataframe(
                        pd.DataFrame(centers, columns=numeric_cols),
                        height=200
                    )

                    # 数据下载
                    st.download_button(
                        label="下载报告",
                        data=generate_report(features, labels, bandwidth),
                        file_name="cluster_report.csv"
                    )

        except Exception as e:
            st.error(f"分析失败：{str(e)}")
            st.code("""
            常见问题排查：
            1. 尝试增大带宽参数
            2. 检查数据是否标准化
            3. 确保没有缺失值
            """)


def generate_report(data, labels, bandwidth):
    """生成分析报告"""
    report = f"""
    均值漂移聚类分析报告
    =====================

    基本信息：
    - 样本数量：{len(data)}
    - 特征维度：{data.shape[1]}
    - 使用带宽：{bandwidth:.2f}
    - 检测到聚类数：{len(np.unique(labels))}

    质量评估：
    - 轮廓系数：{silhouette_score(data, labels):.2f}

    类别分布：
    {pd.Series(labels).value_counts().to_markdown()}
    """
    return report.encode('utf-8')


if __name__ == "__main__":
    main()
