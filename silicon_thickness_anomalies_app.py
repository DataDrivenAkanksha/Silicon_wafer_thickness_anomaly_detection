# silicon_thickness_anomalies_app.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Wafer Thickness Anomaly Detection", layout="wide")
st.title("🧪 Silicon Wafer Thickness Anomaly Detection")

# Sidebar
st.sidebar.header("Settings")
alpha = st.sidebar.select_slider(
    "Significance Level (α)", options=[0.1, 0.05, 0.01, 0.001], value=0.01
)
uploaded_file = st.sidebar.file_uploader("Upload CSV File (9 columns)", type=["csv"])

# Small fallback sample dataset
sample_csv = """
0.175,0.188,-0.159,0.095,0.374,-0.238,-0.8,0.158,-0.211
0.102,0.075,0.141,0.18,0.138,-0.057,-0.075,0.072,0.072
0.607,0.711,0.879,0.765,0.592,0.187,0.431,0.345,0.187
0.774,0.823,0.619,0.37,0.725,0.439,-0.025,-0.259,0.496
0.504,0.644,0.845,0.681,0.502,0.151,0.404,0.296,0.26
"""  # Add more rows as needed

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    st.success("✅ File uploaded successfully!")
else:
    st.info("ℹ️ Using small built-in sample wafer thickness data")
    df = pd.read_csv(io.StringIO(sample_csv.strip()), header=None)

# Clean and validate data
try:
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    if df.shape[1] != 9:
        st.error("❌ Dataset must have exactly 9 columns (for 9 wafer locations).")
        st.stop()
except Exception as e:
    st.error(f"❌ Error processing data: {e}")
    st.stop()

# Rename columns
df.columns = [f"loc{i+1}" for i in range(9)]
df["batch"] = np.arange(1, len(df)+1)

# Mahalanobis distance function
def mahalanobis(x, mean_vec, inv_cov):
    d = x - mean_vec
    return np.sqrt(d @ inv_cov @ d.T)

# Calculate Mahalanobis distances
loc_cols = [f"loc{i+1}" for i in range(9)]
X = df[loc_cols].values
mean_vec = X.mean(axis=0)
cov = np.cov(X, rowvar=False)
inv_cov = np.linalg.pinv(cov)

df["mahal_dist"] = [mahalanobis(x, mean_vec, inv_cov) for x in X]
threshold = np.sqrt(chi2.ppf(1 - alpha, df=len(loc_cols)))
df["is_anomaly"] = df["mahal_dist"] > threshold
anomalies = df[df["is_anomaly"]]

# Preview
st.subheader("📄 Data Preview")
st.dataframe(df.head(), use_container_width=True)

st.success(f"🚨 Detected {len(anomalies)} anomalies (α = {alpha}, threshold ≈ {threshold:.2f})")

# Plot Mahalanobis distance
st.subheader("📊 Mahalanobis Distance per Batch")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["batch"], df["mahal_dist"], label="Distance", marker="o")
ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.2f})")
ax.set_xlabel("Batch")
ax.set_ylabel("Mahalanobis Distance")
ax.set_title("Mahalanobis Distance per Batch")
ax.legend()
st.pyplot(fig)

# Boxplot
st.subheader("📦 Thickness Boxplot by Location")
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.boxplot(data=df[loc_cols], ax=ax2)
st.pyplot(fig2)

# Show anomalies
st.subheader("🚨 Anomalous Batches")
st.dataframe(anomalies[["batch", "mahal_dist"] + loc_cols], use_container_width=True)

# Download anomalies
st.download_button("📥 Download Anomalies CSV", anomalies.to_csv(index=False), "anomalies.csv", "text/csv")


