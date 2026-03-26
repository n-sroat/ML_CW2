import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def generate_cluster_labels(
    embeddings_base,
    max_clusters: int = 500,
    labeled_points: int = 0,
    budget: int = 60,
    random_state: int = 42,
    batch_size: int = 1024,
    verbose: bool = True
):
    embeddings = embeddings_base

    if verbose:
        print("Loaded embeddings:", embeddings.shape)

    K = min(labeled_points + budget, max_clusters)

    if verbose:
        print(f"Using K = {K} clusters")

    if K <= 50:
        if verbose:
            print("Using KMeans")
        cluster_model = KMeans(
            n_clusters=K,
            random_state=random_state
        )
    else:
        if verbose:
            print("Using MiniBatchKMeans")
        cluster_model = MiniBatchKMeans(
            n_clusters=K,
            batch_size=batch_size,
            random_state=random_state
        )

    cluster_labels = cluster_model.fit_predict(embeddings)

    if verbose:
        print("Clustering done. Cluster labels shape:", cluster_labels.shape)
    return cluster_labels