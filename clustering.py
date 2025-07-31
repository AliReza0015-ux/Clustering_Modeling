# clustering.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(filepath):
    return pd.read_csv(filepath)

def train_kmeans(df, features, n_clusters):
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    model.fit(df[features])
    return model

def calculate_wcss(df, features, k_range):
    wcss = []
    for k in k_range:
        kmodel = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmodel.fit(df[features])
        wcss.append(kmodel.inertia_)
    return wcss

def calculate_silhouette(df, features, k_range):
    scores = []
    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = model.fit_predict(df[features])
        score = silhouette_score(df[features], labels)
        scores.append(score)
    return scores
