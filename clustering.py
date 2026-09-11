import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def calculate_silhouette_scores(X_scaled, min_k=2, max_k=10):
    """Calculate silhouette scores for a range of cluster counts."""
    scores = []

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)

    return scores


def get_optimal_k(scores, min_k=2):
    """Return the cluster count with the highest silhouette score."""
    return min_k + scores.index(max(scores))


def train_kmeans(X_scaled, n_clusters):
    """Train the final K-Means clustering model."""
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)
    return model, labels


def create_cluster_profiles(df):
    """Create summary statistics for each customer cluster."""
    profiles = df.groupby("Cluster").agg(
        Customer_Count=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending_Score=("Spending Score (1-100)", "mean")
    ).reset_index()

    profiles["Average_Age"] = profiles["Average_Age"].round(1)
    profiles["Average_Income"] = profiles["Average_Income"].round(1)
    profiles["Average_Spending_Score"] = profiles[
        "Average_Spending_Score"
    ].round(1)

    return profiles


def assign_segment_name(row):
    """Assign a descriptive business-friendly name to a cluster."""
    income = row["Average_Income"]
    spending = row["Average_Spending_Score"]

    if income >= 60 and spending >= 60:
        return "High-Value Customers"
    elif income >= 60 and spending < 40:
        return "High-Income Low-Spenders"
    elif income < 40 and spending >= 60:
        return "Budget High-Spenders"
    elif income < 40 and spending < 40:
        return "Budget Low-Spenders"
    else:
        return "Average Customers"
