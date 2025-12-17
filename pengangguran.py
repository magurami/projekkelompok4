import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset_2019.csv")

X = df.drop(['Kabupaten'], axis=1)


st.header("Isi dataset")
st.write(X)

clusters=[]
for i in range (1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)


st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih Jumlah Cluster :", 2, 10, 3, 1)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, marker='o', ax=ax)

ax.set_title('Mencari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')


elbow_k = clust
elbow_y = clusters[elbow_k - 1]

ax.annotate(
    f'K = {elbow_k}',
    xy=(elbow_k, elbow_y),
    xytext=(elbow_k + 1, elbow_y + 0.05 * max(clusters)),
    arrowprops=dict(arrowstyle='->', lw=2)
)


ax.scatter(elbow_k, elbow_y, s=100, zorder=5)

st.pyplot(fig)

features = ["SMP", "SMA", "DIPLOMA", "Jumlah_Pengangguran"]

x_feature = st.sidebar.selectbox("Pilih Sumbu X", features, index=0)
y_feature = st.sidebar.selectbox("Pilih Sumbu Y", features, index=1)

n_clust = st.sidebar.slider("Jumlah Cluster", 2, 10, 3)
st.markdown("""
**Elbow Method** digunakan untuk menentukan jumlah cluster optimal.
Titik siku menunjukkan nilai K terbaik di mana penurunan inertia mulai melambat.
""")

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=n_clust, random_state=42)
df["Labels"] = kmeans.fit_predict(scaled_features)

fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    data=df,
    x=x_feature,
    y=y_feature,
    hue="Labels",
    palette="tab10",
    s=80,
    ax=ax
)

# Tulis label di titik rata-rata cluster
for label in df["Labels"].unique():
    x_mean = df[df["Labels"] == label][x_feature].mean()
    y_mean = df[df["Labels"] == label][y_feature].mean()

    ax.annotate(
        f"Cluster {label}",
        (x_mean, y_mean),
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color="black"
    )

ax.set_title("Visualisasi K-Means Clustering")
st.pyplot(fig)
