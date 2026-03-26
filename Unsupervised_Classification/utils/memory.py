"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim

        # safer initialization
        self.features = torch.zeros(self.n, self.dim, dtype=torch.float32)
        self.targets = torch.zeros(self.n, dtype=torch.long)

        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        """
        Perform weighted k-NN using the memory bank.
        Both predictions and memory bank features are on CPU.
        """
        # Ensure predictions are on the same device as memory bank (CPU)
        predictions = predictions.cpu()
        
        batch_size = predictions.shape[0]

        # Prepare one-hot tensor for retrieved neighbors
        retrieval_one_hot = torch.zeros(batch_size * self.K, self.C, dtype=torch.float32)

        # Compute correlation with memory bank features (CPU)
        correlation = torch.matmul(predictions, self.features.t())

        # Get top-K nearest neighbors
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)

        # Gather targets of neighbors
        candidates = self.targets.view(1, -1).expand(batch_size, -1)
        retrieval = torch.gather(candidates, 1, yi)

        # Fill one-hot matrix
        retrieval_one_hot.zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)

        # Weighted probabilities (avoid inplace operations)
        yd_transform = torch.exp(yd / self.temperature)
        probs = torch.sum(
            retrieval_one_hot.view(batch_size, -1, self.C) *
            yd_transform.view(batch_size, -1, 1),
            dim=1
        )

        # Get predicted class
        _, class_preds = probs.sort(1, descending=True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        """
        Perform standard k-NN using the memory bank.
        Assumes memory bank features and predictions are on CPU.
        """
        # Ensure predictions are on CPU
        predictions = predictions.cpu()

        # Compute correlations
        correlation = torch.matmul(predictions, self.features.t())

        # Get nearest neighbor index
        sample_pred = torch.argmax(correlation, dim=1)

        # Map to class labels
        class_pred = torch.index_select(self.targets, 0, sample_pred)

        return class_pred


    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        """
        Mine top-k nearest neighbors for every sample using FAISS (CPU only).
        """
        import faiss

        # Use CPU features
        features = self.features.cpu().numpy().astype('float32')
        n, dim = features.shape

        # FAISS CPU index
        index = faiss.IndexFlatIP(dim)
        index.add(features)

        # Search top-k+1 (including self)
        distances, indices = index.search(features, topk + 1)

        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # exclude self
            anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        else:
            return indices


    def reset(self):
        self.ptr = 0


    def update(self, features, targets):
        """
        Update memory bank with new features and targets.
        Always move to CPU to avoid hidden CUDA syncs.
        """
        b = features.size(0)
        assert (b + self.ptr <= self.n)

        self.features[self.ptr:self.ptr + b].copy_(features.detach().cpu())
        self.targets[self.ptr:self.ptr + b].copy_(targets.detach().cpu())
        self.ptr += b


    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device


    def cpu(self):
        self.to('cpu')


    def cuda(self):
        self.to('cuda:0')