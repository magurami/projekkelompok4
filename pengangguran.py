import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="K-Means Pengangguran", layout="wide")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("dataset_2019.csv")

features = ["SMP", "SMA", "DIPLOMA", "Jumlah_Pengangguran"]

# ===============================
# SIDEBAR MENU
# ===============================
menu = st.sidebar.radio(
    "Menu",
    ["EDA (Exploratory Data Analysis)", "K-Means Clustering"]
)

# ===============================
# EDA MENU
# ===============================
if menu == "EDA (Exploratory Data Analysis)":

    st.header("📊 Exploratory Data Analysis (EDA)")

    st.subheader("📄 Tampilan Dataset")
    st.dataframe(df)

    st.subheader("ℹ️ Informasi Dataset")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Jumlah Baris:", df.shape[0])
        st.write("Jumlah Kolom:", df.shape[1])

    with col2:
        st.write("Nama Kolom:")
        st.write(df.columns.tolist())

    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df[features].describe())

    st.subheader("📊 Distribusi Fitur")

    fitur_pilih = st.selectbox("Pilih Fitur", features)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[fitur_pilih], bins=15, kde=True, ax=ax)
    ax.set_title(f"Distribusi {fitur_pilih}")
    st.pyplot(fig)

    st.subheader("🔗 Korelasi Antar Fitur")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[features].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Heatmap Korelasi")
    st.pyplot(fig)

# ===============================
# K-MEANS MENU
# ===============================
else:

    st.header("🔍 K-Means Clustering")

    X = df[features]

    st.subheader("📄 Data yang Digunakan untuk Clustering")
    st.dataframe(X)

    # ===============================
    # ELBOW METHOD
    # ===============================
    clusters = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X)
        clusters.append(km.inertia_)

    st.sidebar.subheader("⚙️ Elbow Method")
    clust = st.sidebar.slider("Pilih Jumlah Cluster (K)", 2, 10, 3)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=list(range(1, 11)), y=clusters, marker="o", ax=ax)

    elbow_y = clusters[clust - 1]
    ax.scatter(clust, elbow_y, s=120)
    ax.annotate(
        f"K = {clust}",
        xy=(clust, elbow_y),
        xytext=(clust + 0.5, elbow_y),
        arrowprops=dict(arrowstyle="->")
    )

    ax.set_title("Elbow Method")
    ax.set_xlabel("Jumlah Cluster")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

    st.markdown("""
    **Elbow Method** digunakan untuk menentukan jumlah cluster optimal.  
    Titik siku menunjukkan nilai K terbaik saat penurunan inertia mulai melambat.
    """)

    # ===============================
    # K-MEANS MODEL
    # ===============================
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=clust, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_features)

    # ===============================
    # VISUALISASI CLUSTER
    # ===============================
    st.subheader("📌 Visualisasi Cluster")

    x_feature = st.selectbox("Pilih Sumbu X", features, index=0)
    y_feature = st.selectbox("Pilih Sumbu Y", features, index=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x=x_feature,
        y=y_feature,
        hue="Cluster",
        palette="tab10",
        s=80,
        ax=ax
    )

    for label in df["Cluster"].unique():
        x_mean = df[df["Cluster"] == label][x_feature].mean()
        y_mean = df[df["Cluster"] == label][y_feature].mean()
        ax.annotate(
            f"Cluster {label}",
            (x_mean, y_mean),
            ha="center",
            va="center",
            fontsize=12,
            weight="bold"
        )

    ax.set_title("Visualisasi Hasil K-Means")
    st.pyplot(fig)

    # ===============================
    # DATASET HASIL CLUSTER
    # ===============================
    st.subheader("📄 Dataset Setelah Clustering")
    st.dataframe(df)

    st.success("Setiap baris sekarang sudah memiliki label cluster 🎯")
