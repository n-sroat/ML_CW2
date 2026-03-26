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
from typiclust_modification import select_typiclust_points_dynamic, decide_strategy

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

    budgets = [100,200,300,400,500,600]
    #budgets = [500,600,700,800,900,1000]

    medoid_mean = []
    tpcrp_mean_means = []
    tpcrp_mean_base = []

    past_eval_accuracies = []

    labeled_set_dynamic = set()
    labeled_set_base = set()

    use_hybrid = True
    previous_budget = 0

    for budget in budgets:
        print(f"\nBudget {budget}")

        increment = budget - previous_budget

        cluster_labels = generate_cluster_labels(
            embeddings_base=train_embeddings_base,
            budget=budget
        )

        # Decide strategy BEFORE selection
        use_hybrid = decide_strategy(
            eval_accuracies=past_eval_accuracies,
            current_strategy=use_hybrid,
            threshold_ratio=0.15
        )

        # Dynamic switched selection
        new_points_dynamic = select_typiclust_points_dynamic(
            embeddings=train_embeddings,
            cluster_labels=cluster_labels,
            labeled_set=labeled_set_dynamic,
            budget=increment,
            use_hybrid=use_hybrid,
            k_neighbors=20
        )

        tpcrp_indices = list(labeled_set_dynamic)

        tpcrp_acc = linear_eval(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            selected_indices=tpcrp_indices,
            test_embeddings=test_embeddings,
            test_labels=test_labels
        )

        past_eval_accuracies.append(tpcrp_acc)
        tpcrp_mean_means.append(tpcrp_acc)


        new_points_base = select_typiclust_points(
            embeddings=train_embeddings,
            cluster_labels=cluster_labels,
            labeled_set=labeled_set_base,
            budget=increment,
            k_neighbors=20,
        )

        tpcrp_indices_base = list(labeled_set_base)

        tpcrp_acc_base = linear_eval(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            selected_indices=tpcrp_indices_base,
            test_embeddings=test_embeddings,
            test_labels=test_labels
        )

        tpcrp_mean_base.append(tpcrp_acc_base)

        previous_budget = budget

    plt.figure(figsize=(8, 6))


    plt.plot(
        budgets,
        tpcrp_mean_means,
        marker='s',
        label='TypiClust Switch'
    )

    plt.plot(
        budgets,
        tpcrp_mean_base,
        marker='^',
        label='TypiClust Base'
    )

    plt.xlabel("Budget")
    plt.ylabel("Accuracy")
    plt.title("Linear Eval Accuracy vs Budget")
    plt.legend()
    plt.grid(True)

    plt.savefig("modified_hundreds.pdf")
    print("Figure saved as accuracy_vs_budget.pdf")
    plt.show()