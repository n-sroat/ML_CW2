import numpy as np 
import torch
import matplotlib.pyplot as plt

from typiclust_modification import select_typiclust_points_dynamic, decide_strategy
from step2 import generate_cluster_labels
from typiclust import select_typiclust_points
from linear import linear_eval


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