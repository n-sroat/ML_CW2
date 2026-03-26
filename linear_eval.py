import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from torchvision import datasets, transforms
from typiclust import select_typiclust_points
from step2 import generate_cluster_labels
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
from sampling import k_center_greedy

def linear_eval(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    selected_indices: List[int],
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int = 10,
    num_epochs: int = 150,
    lr: float = 2.0,
    momentum: float = 0.9,
    use_random: bool = False,
    random_seed: int = 0,
    return_predictions: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:

    if use_random:
        torch.manual_seed(random_seed)
        selected_indices = torch.randperm(len(train_embeddings))[:len(selected_indices)]

    X_train = train_embeddings[selected_indices]
    y_train = train_labels[selected_indices]

    linear_layer = nn.Linear(X_train.shape[1], num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(linear_layer.parameters(), lr=lr, momentum=momentum)

    for _ in range(num_epochs):
        linear_layer.train()
        optimizer.zero_grad()
        loss = criterion(linear_layer(X_train), y_train)
        loss.backward()
        optimizer.step()

    linear_layer.eval()
    with torch.no_grad():
        test_outputs = linear_layer(test_embeddings)
        _, test_pred = torch.max(test_outputs, 1)
        test_acc = (test_pred == test_labels).float().mean().item()

    if return_predictions:
        return test_acc, test_pred.cpu().numpy()
    else:
        return test_acc


if __name__ == "__main__":
    train_embeddings_base = np.load("cifar10_embeddings_train.npy")
    train_embeddings = torch.tensor(train_embeddings_base, dtype=torch.float32)
    test_embeddings = torch.tensor(np.load("cifar10_embeddings_test.npy"), dtype=torch.float32)

    train_labels = torch.tensor(np.load("cifar10_labels_train.npy"), dtype=torch.long)
    test_labels = torch.tensor(np.load("cifar10_labels_test.npy"), dtype=torch.long)

    budgets = [10,20,30,40,50,60]  

    num_repeats = 10  

    random_means, random_stds = [], []
    coreset_means, coreset_stds = [], []
    tpcrp_mean_means, tpcrp_mean_stds = [], []

    for budget in budgets:
        print(f"\nBudget {budget}")

        labeled_set = set()
        cluster_labels = generate_cluster_labels(
            embeddings_base=train_embeddings_base,
            budget=budget
        )

        tpcrp_indices = select_typiclust_points(
            embeddings=train_embeddings,
            cluster_labels=cluster_labels,
            labeled_set=labeled_set,
            budget=budget,
            k_neighbors=20
        )

        tpcrp_acc = linear_eval(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            selected_indices=tpcrp_indices,
            test_embeddings=test_embeddings,
            test_labels=test_labels
        )

        tpcrp_mean_means.append(tpcrp_acc)
        tpcrp_mean_stds.append(0)  # deterministic

        # ----------- Random ----------- 
        random_accuracies = []
        for seed in range(num_repeats):
            np.random.seed(seed)
            random_indices = np.random.choice(len(train_embeddings), budget, replace=False)
            acc = linear_eval(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                selected_indices=random_indices,
                test_embeddings=test_embeddings,
                test_labels=test_labels
            )
            random_accuracies.append(acc)

        random_means.append(np.mean(random_accuracies))
        random_stds.append(np.std(random_accuracies))

        # ----------- Coreset ----------- 
        coreset_accuracies = []
        for seed in range(num_repeats):
            np.random.seed(seed)
            coreset_indices = k_center_greedy(train_embeddings_base, budget)
            acc = linear_eval(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                selected_indices=coreset_indices,
                test_embeddings=test_embeddings,
                test_labels=test_labels
            )
            coreset_accuracies.append(acc)

        coreset_means.append(np.mean(coreset_accuracies))
        coreset_stds.append(np.std(coreset_accuracies))

        print(f"Random: {random_means[-1]:.4f} ± {random_stds[-1]:.4f}")
        print(f"Coreset: {coreset_means[-1]:.4f} ± {coreset_stds[-1]:.4f}")
        print(f"TypiClust: {tpcrp_mean_means[-1]:.4f}")


    plt.figure(figsize=(8, 6))

    # Random
    plt.errorbar(
        budgets,
        random_means,
        yerr=random_stds,
        marker='o',
        label='Random'
    )

    plt.errorbar(
        budgets,
        coreset_means,
        yerr=coreset_stds,
        marker='^',
        label='Coreset'
    )

    # TypiClust
    plt.plot(
        budgets,
        tpcrp_mean_means,
        marker='s',
        label='TypiClust'
    )

    plt.xlabel("Budget")
    plt.ylabel("Accuracy")
    plt.title("Linear Eval Accuracy vs Budget")
    plt.legend()
    plt.grid(True)

    # Save as PDF
    plt.savefig("accuracy_vs_budget.pdf")
    print("Figure saved as accuracy_vs_budget.pdf")
    plt.show()