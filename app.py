# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
from clustering import load_data, train_kmeans, calculate_wcss, calculate_silhouette

st.set_page_config(page_title="Customer Clustering", layout="wide")
st.title("🧠 Mall Customer Segmentation (KMeans Clustering)")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data("data/mall_customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

features = st.multiselect("Select features for clustering", df.select_dtypes(include='number').columns.tolist(),
                          default=['Annual_Income', 'Spending_Score'])

k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=5)

model = train_kmeans(df, features, k)
df['Cluster'] = model.labels_

st.subheader("📊 Cluster Visualization")
if len(features) == 2:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=features[0], y=features[1], hue='Cluster', palette='Set2', ax=ax)
    st.pyplot(fig)
else:
    st.warning("Select exactly 2 features to see scatter plot.")

# Elbow plot
st.subheader("📈 Elbow Method")
k_range = range(2, 11)
wcss = calculate_wcss(df, features, k_range)
fig2, ax2 = plt.subplots()
ax2.plot(k_range, wcss, marker='o')
ax2.set_xlabel("k")
ax2.set_ylabel("WCSS")
ax2.set_title("Elbow Plot")
st.pyplot(fig2)

# Silhouette plot
st.subheader("📈 Silhouette Score")
sil = calculate_silhouette(df, features, k_range)
fig3, ax3 = plt.subplots()
ax3.plot(k_range, sil, marker='o', color='green')
ax3.set_xlabel("k")
ax3.set_ylabel("Silhouette Score")
ax3.set_title("Silhouette Plot")
st.pyplot(fig3)
