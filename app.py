import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

st.set_page_config(layout="wide")
st.title("🌍 World Development Clustering Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")

    # Initial cleaning
    for col in ['Business Tax Rate', 'GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('%', '').str.replace('$', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.drop(columns=['Ease of Business'], inplace=True, errors='ignore')

    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encoding and scaling
    le = LabelEncoder()
    df['country_encoded'] = le.fit_transform(df['Country'])
    df_countries = df['Country']
    df.drop('Country', axis=1, inplace=True)

    numerical_columns = df.select_dtypes(['int64', 'float64']).columns
    rs = RobustScaler()
    df_rs = df.copy()
    df_rs[numerical_columns] = rs.fit_transform(df[numerical_columns])

    # PCA
    pca = PCA(n_components=0.95)
    pca_data = pca.fit_transform(df_rs)
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])

    st.subheader("📊 PCA Explained Variance")
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='green')
    ax.set_title('Cumulative Explained Variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Variance')
    st.pyplot(fig)

    # Elbow method
    inertia = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pca_df)
        inertia.append(kmeans.inertia_)

    st.subheader("📐 Elbow Curve for KMeans")
    fig, ax = plt.subplots()
    ax.plot(K, inertia, marker='o')
    ax.set_title("Elbow Curve")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(df_rs)

    st.subheader("📌 KMeans Clustering (K=3)")
    fig, ax = plt.subplots()
    ax.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    ax.set_title("KMeans Clusters")
    st.pyplot(fig)

    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=3, linkage='ward')
    agglo_labels = agglo.fit_predict(df_rs)

    st.subheader("🔗 Agglomerative Clustering")
    fig, ax = plt.subplots()
    ax.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c=agglo_labels, cmap='rainbow', s=50)
    ax.set_title("Agglomerative Clusters")
    st.pyplot(fig)

    # Dendrogram
    st.subheader("🌲 Dendrogram")
    fig, ax = plt.subplots(figsize=(10, 6))
    sch.dendrogram(sch.linkage(pca_df, method='ward'), ax=ax)
    ax.axhline(y=4000, color='r', linestyle='--')
    st.pyplot(fig)

    # DBSCAN
    best_score = -1
    best_eps = None
    for eps in [20.0, 25.0, 30.0, 40.0]:
        dbscan = DBSCAN(eps=eps, min_samples=10)
        labels = dbscan.fit_predict(df_rs)
        if len(set(labels)) > 1:
            score = silhouette_score(df_rs, labels)
            if score > best_score:
                best_score = score
                best_eps = eps

    dbscan = DBSCAN(eps=best_eps, min_samples=10)
    db_labels = dbscan.fit_predict(df_rs)

    st.subheader("🌀 DBSCAN Clustering")
    fig, ax = plt.subplots()
    ax.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c=db_labels, cmap='rainbow', s=50)
    ax.set_title("DBSCAN Clusters")
    st.pyplot(fig)

    # Silhouette Scores
    st.subheader("📈 Silhouette Scores")
    kmeans_score = silhouette_score(df_rs, kmeans_labels)
    agglo_score = silhouette_score(df_rs, agglo_labels)
    st.write(f"**KMeans Score:** {kmeans_score:.3f}")
    st.write(f"**Agglomerative Score:** {agglo_score:.3f}")
    st.write(f"**DBSCAN Score:** {best_score:.3f} (eps={best_eps})")
