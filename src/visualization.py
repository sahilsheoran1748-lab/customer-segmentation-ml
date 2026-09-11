import matplotlib.pyplot as plt


def plot_elbow_curve(k_values, inertia_values, output_path):
    """Create and save the K-Means elbow curve."""
    plt.figure(figsize=(9, 6))
    plt.plot(k_values, inertia_values, marker="o")
    plt.title("Elbow Curve for K-Means Clustering")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_silhouette_scores(k_values, scores, output_path):
    """Create and save the silhouette score analysis."""
    plt.figure(figsize=(9, 6))
    plt.plot(k_values, scores, marker="o")
    plt.title("Silhouette Score Analysis")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_customer_clusters(
    df,
    model,
    scaler,
    output_path
):
    """Create and save the customer cluster visualization."""
    plt.figure(figsize=(10, 7))

    for cluster in sorted(df["Cluster"].unique()):
        cluster_data = df[df["Cluster"] == cluster]

        plt.scatter(
            cluster_data["Annual Income (k$)"],
            cluster_data["Spending Score (1-100)"],
            label=cluster_data["Customer_Segment"].iloc[0],
            alpha=0.75,
            s=55
        )

    centers_original = scaler.inverse_transform(
        model.cluster_centers_
    )

    plt.scatter(
        centers_original[:, 0],
        centers_original[:, 1],
        marker="X",
        s=220,
        edgecolors="black",
        linewidths=1.5,
        label="Cluster Centers"
    )

    plt.title("Customer Segmentation using K-Means")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
