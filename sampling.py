import torch
from typing import List

import numpy as np

def k_center_greedy(embeddings, coreset_size):
    n = embeddings.shape[0]
    # Start with a random point
    indices = [np.random.randint(n)]

    # Track distances to nearest selected point
    distances = np.linalg.norm(embeddings - embeddings[indices[0]], axis=1)

    for _ in range(1, coreset_size):
        # Select the point farthest from current coreset
        next_index = np.argmax(distances)
        indices.append(next_index)
        # Update distances
        dist_to_new = np.linalg.norm(embeddings - embeddings[next_index], axis=1)
        distances = np.minimum(distances, dist_to_new)

    return np.array(indices)

'''# Example usage:
# embeddings = np.load("your_embeddings.npy")  # shape (num_samples, embedding_dim)
coreset_indices = k_center_greedy(embeddings, coreset_size=100)
coreset_embeddings = embeddings[coreset_indices]'''

def select_random_points(
    dataset_size: int,
    budget: int,
    seed: int
) -> List[int]:

    torch.manual_seed(seed)

    indices = torch.randperm(dataset_size)[:budget]

    return indices.tolist()


if __name__ == "__main__":

    dataset_size = 50000   # CIFAR-10 train size
    budget = 10

    seeds = [0, 1, 2, 3, 4]

    for seed in seeds:
        selected = select_random_points(
            dataset_size=dataset_size,
            budget=budget,
            seed=seed
        )

        print(f"Seed {seed}: {selected}")