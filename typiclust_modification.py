import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def compute_dynamic_hybrid_typicality(cluster_embeddings, 
                                      k_fraction=0.2,      # fraction of cluster to use for k
                                      k_max=20,            # maximum k to avoid very large neighborhoods
                                      eps=1e-8, 
                                      cluster_threshold=15):

    n = len(cluster_embeddings)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones(1)

    # Dynamic alpha: more local weight for small clusters
    alpha = max(0.0, min(1.0, 1.0 - (n - 1) / (cluster_threshold - 1)))

    # Adapt k based on cluster size
    k = max(1, min(k_max, int(n * k_fraction)))
    if k >= n:
        k = n - 1

    # Local k-NN density
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(cluster_embeddings)
    distances, _ = nbrs.kneighbors(cluster_embeddings)
    distances_no_self = distances[:, 1:]
    avg_dist = distances_no_self.mean(axis=1)
    local_score = 1 / (avg_dist + eps)

    # Global centrality (medoid-style)
    dist_matrix = pairwise_distances(cluster_embeddings)
    total_distances = dist_matrix.sum(axis=1)
    global_score = 1 / (total_distances + eps)

    # Hybrid score
    hybrid_score = alpha * local_score + (1 - alpha) * global_score
    return hybrid_score

def k_center_greedy_scores(cluster_embeddings):
    n = cluster_embeddings.shape[0]

    if n == 0:
        return np.array([])

    indices = [np.random.randint(n)]
    distances = np.linalg.norm(cluster_embeddings - cluster_embeddings[indices[0]], axis=1)

    for _ in range(1, n):
        next_index = np.argmax(distances)
        indices.append(next_index)

        dist_to_new = np.linalg.norm(cluster_embeddings - cluster_embeddings[next_index], axis=1)
        distances = np.minimum(distances, dist_to_new)

    scores = np.zeros(n)
    scores[indices] = np.arange(n, 0, -1)

    return scores


def decide_strategy(eval_accuracies, current_strategy=True, threshold_ratio=0.10):
    """
    True = hybrid
    False = coreset
    """

    if eval_accuracies is None or len(eval_accuracies) < 5:
        return current_strategy

    gains = np.diff(eval_accuracies)

    baseline = np.mean(gains[:2])
    threshold = baseline * threshold_ratio

    recent_mean = np.mean(gains[-3:])

    if recent_mean < threshold:
        return not current_strategy

    return current_strategy


def select_typiclust_points_dynamic(
    embeddings,
    cluster_labels,
    labeled_set,
    budget,
    use_hybrid=True,
    min_cluster_size=5,
    k_neighbors=20,
):
    selected_points = []

    clusters = {}
    for i, c in enumerate(cluster_labels):
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(i)

    clusters = {c: pts for c, pts in clusters.items() if len(pts) >= min_cluster_size}

    cluster_unlabeled = {}
    for c in clusters:
        cluster_unlabeled[c] = [p for p in clusters[c] if p not in labeled_set]

    while len(selected_points) < budget:

        if len(cluster_unlabeled) == 0:
            break

        labeled_counts = {
            c: len(clusters[c]) - len(cluster_unlabeled[c])
            for c in cluster_unlabeled
        }

        min_labeled = min(labeled_counts.values())

        candidate_clusters = [
            c for c in labeled_counts if labeled_counts[c] == min_labeled
        ]

        chosen_cluster = max(candidate_clusters, key=lambda c: len(cluster_unlabeled[c]))

        points_in_cluster = cluster_unlabeled[chosen_cluster]

        if len(points_in_cluster) == 0:
            del cluster_unlabeled[chosen_cluster]
            continue

        cluster_embeddings = embeddings[points_in_cluster]

        if use_hybrid:
            scores = compute_dynamic_hybrid_typicality(
                cluster_embeddings,
            )
        else:
            scores = k_center_greedy_scores(cluster_embeddings)

        idx_max = np.argmax(scores)
        selected_point = points_in_cluster[idx_max]

        selected_points.append(selected_point)
        labeled_set.add(selected_point)

        cluster_unlabeled[chosen_cluster].remove(selected_point)

        if len(cluster_unlabeled[chosen_cluster]) == 0:
            del cluster_unlabeled[chosen_cluster]

    return selected_points