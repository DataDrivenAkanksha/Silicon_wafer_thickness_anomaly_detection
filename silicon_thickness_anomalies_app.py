# streamlit_wafer_anomaly.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Wafer Thickness Anomaly Detection", layout="wide")
st.title("🧪 Silicon Wafer Thickness Anomaly Detection")

# Sidebar
st.sidebar.header("Settings")
alpha = st.sidebar.select_slider(
    "Significance Level (α)", options=[0.1, 0.05, 0.01, 0.001], value=0.01
)
uploaded_file = st.sidebar.file_uploader("Upload CSV File (9 columns)", type=["csv"])

# Load data (either uploaded or sample)
if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    st.success("✅ File uploaded successfully!")
else:
    st.info("ℹ️ Using sample wafer thickness dataset (OpenMV)")
    url = "https://openmv.net/file/silicon-wafer-thickness.csv"
    df = pd.read_csv(url, header=None)

# Process Data
if df is not None and not df.empty:
    df.columns = [f"loc{i+1}" for i in range(9)]
    df["batch"] = np.arange(1, len(df)+1)

    # Mahalanobis
    def mahalanobis(x, mean_vec, inv_cov):
        d = x - mean_vec
        return np.sqrt(d @ inv_cov @ d.T)

    loc_cols = [f"loc{i+1}" for i in range(9)]
    X = df[loc_cols].values
    mean_vec = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.inv(cov)

    df["mahal_dist"] = [mahalanobis(x, mean_vec, inv_cov) for x in X]
    threshold = np.sqrt(chi2.ppf(1 - alpha, df=len(loc_cols)))
    df["is_anomaly"] = df["mahal_dist"] > threshold
    anomalies = df[df["is_anomaly"]]

    st.subheader("📄 Preview of Dataset")
    st.dataframe(df.head(), use_container_width=True)

    st.success(f"🚨 Detected {len(anomalies)} anomalies (α = {alpha})")

    # Plot distances
    st.subheader("📊 Mahalanobis Distance per Batch")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["batch"], df["mahal_dist"], label="Distance", marker="o")
    ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.2f})")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Mahalanobis Distance")
    ax.set_title("Mahalanobis Distance vs Batch")
    ax.legend()
    st.pyplot(fig)

    # Boxplot of locations
    st.subheader("📦 Thickness by Location")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df[loc_cols], ax=ax2)
    ax2.set_title("Wafer Thickness per Measurement Location")
    st.pyplot(fig2)

    # Show anomalies
    st.subheader("🚨 Anomalous Batches")
    st.dataframe(anomalies[["batch", "mahal_dist"] + loc_cols], use_container_width=True)

    # Download
    st.download_button("📥 Download Anomalies CSV", anomalies.to_csv(index=False), "anomalies.csv", "text/csv")
else:
    st.error("❌ No data loaded.")

