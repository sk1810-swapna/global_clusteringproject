import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

st.set_page_config(layout="wide")
st.title("🔗 Agglomerative Clustering on World Development Data")

# Sidebar for cluster selection
st.sidebar.header("🔧 Clustering Settings")
n_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=3)

# Upload file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")

    # Clean and preprocess
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

    # Encode and scale
    le = LabelEncoder()
    df['country_encoded'] = le.fit_transform(df['Country'])
    df_countries = df['Country']
    df.drop('Country', axis=1, inplace=True)

    numerical_columns = df.select_dtypes(['int64', 'float64']).columns
    rs = RobustScaler()
    df_scaled = df.copy()
    df_scaled[numerical_columns] = rs.fit_transform(df[numerical_columns])

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])

    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agglo_labels = agglo.fit_predict(df_scaled)

    # Silhouette Score
    score = silhouette_score(df_scaled, agglo_labels)

    # Add cluster labels to data
    df_result = df.copy()
    df_result['Country'] = df_countries
    df_result['Cluster'] = agglo_labels
    pca_df['Cluster'] = agglo_labels
    pca_df['Country'] = df_countries

    st.subheader(f"📌 Agglomerative Clustering Visualization (n={n_clusters})")
    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=60, ax=ax)
    ax.set_title(f"Agglomerative Clusters (Silhouette Score = {score:.2f})")
    st.pyplot(fig)

    st.subheader("📋 Clustered Country Table")
    st.dataframe(df_result[['Country', 'Cluster']].sort_values('Cluster'))

    st.subheader("📊 Cluster Profiles (Feature Means)")
    cluster_profiles = df_result.groupby('Cluster')[numerical_columns].mean().round(2)
    st.dataframe(cluster_profiles)

    st.success("✅ Clustering complete using Agglomerative Clustering!")
