import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

st.title("🔧 AutoEncoder 降维")

uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("请上传包含至少两个数值特征的CSV文件")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols])

        # Autoencoder model
        input_dim = X_scaled.shape[1]
        encoding_dim = 2  # 目标降维的维度

        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = tf.keras.Model(input_layer, decoded)
        encoder = tf.keras.Model(input_layer, encoded)

        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, shuffle=True)

        X_encoded = encoder.predict(X_scaled)

        st.write("AutoEncoder降维结果：")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_encoded[:, 0], X_encoded[:, 1])
        ax.set_xlabel("维度1")
        ax.set_ylabel("维度2")
        ax.set_title("AutoEncoder 降维结果")
        st.pyplot(fig)
else:
    st.info("请上传一个CSV文件。")
