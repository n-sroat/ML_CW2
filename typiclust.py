import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_typicality(cluster_embeddings, k_neighbors=20):
    n = len(cluster_embeddings)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones(1)

    k = k_neighbors
    if k >= n:
        k = n - 1

    nbrs = NearestNeighbors(n_neighbors=k+1)
    nbrs.fit(cluster_embeddings)
    distances, _ = nbrs.kneighbors(cluster_embeddings)

    # remove self-distance manually
    distances_no_self = distances[:, 1:]
    
    avg_dist = distances_no_self.mean(axis=1)
    typicality_scores = 1 / (avg_dist + 1e-8)
    return typicality_scores


def select_typiclust_points(
    embeddings,
    cluster_labels,
    labeled_set,
    budget,
    min_cluster_size=5,
    k_neighbors=20,
):
    selected_points = []

    # make clusters dict
    clusters = {}
    for i, c in enumerate(cluster_labels):
        if c in clusters:
            clusters[c].append(i)
        else:
            clusters[c] = [i]

    # remove small clusters
    big_clusters = {}
    for c in clusters:
        if len(clusters[c]) >= min_cluster_size:
            big_clusters[c] = clusters[c]
    clusters = big_clusters

    # keep track of which points are unlabeled
    cluster_unlabeled = {}
    for c in clusters:
        pts = []
        for p in clusters[c]:
            if p not in labeled_set:
                pts.append(p)
        cluster_unlabeled[c] = pts

    while len(selected_points) < budget:
        if len(cluster_unlabeled) == 0:
            break

        # count labeled points per cluster
        labeled_counts = {}
        for c in cluster_unlabeled:
            total = len(clusters[c])
            unlabeled = len(cluster_unlabeled[c])
            labeled_counts[c] = total - unlabeled

        # find clusters with min labeled points
        min_labeled = None
        for c in labeled_counts:
            if min_labeled is None:
                min_labeled = labeled_counts[c]
            else:
                if labeled_counts[c] < min_labeled:
                    min_labeled = labeled_counts[c]

        candidate_clusters = []
        for c in labeled_counts:
            if labeled_counts[c] == min_labeled:
                candidate_clusters.append(c)

        # pick largest cluster among candidates
        max_size = -1
        chosen_cluster = None
        for c in candidate_clusters:
            if len(cluster_unlabeled[c]) > max_size:
                max_size = len(cluster_unlabeled[c])
                chosen_cluster = c

        points_in_cluster = cluster_unlabeled[chosen_cluster]
        if len(points_in_cluster) == 0:
            del cluster_unlabeled[chosen_cluster]
            continue

        cluster_embeddings = embeddings[points_in_cluster]
        typicalities = compute_typicality(cluster_embeddings, k_neighbors=k_neighbors)
        typicalities = np.array(typicalities)

        if len(typicalities) != len(points_in_cluster):
            typicalities = np.ones(len(points_in_cluster))

        # pick point with highest typicality
        idx_max = np.argmax(typicalities)
        selected_point = points_in_cluster[idx_max]

        labeled_set.add(selected_point)
        selected_points.append(selected_point)

        # remove from cluster_unlabeled
        cluster_unlabeled[chosen_cluster].remove(selected_point)
        if len(cluster_unlabeled[chosen_cluster]) == 0:
            del cluster_unlabeled[chosen_cluster]

    return selected_points