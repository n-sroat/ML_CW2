import numpy as np
import torch
import matplotlib.pyplot as plt

from step2 import generate_cluster_labels
from typiclust import select_typiclust_points
from typiclust_modification import select_typiclust_points_dynamic, decide_strategy
from resnet import resnet_eval

if __name__ == "__main__":
    train_embeddings_base = np.load("cifar10_embeddings_train.npy")
    train_embeddings = torch.tensor(train_embeddings_base, dtype=torch.float32)

    budgets = [100, 200, 300, 400, 500, 600]

    tpcrp_mean_dynamic = []
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

        # Decide strategy before selection
        use_hybrid = decide_strategy(
            eval_accuracies=past_eval_accuracies,
            current_strategy=use_hybrid,
            threshold_ratio=0.15
        )

        # Dynamic TypiClust selection
        new_points_dynamic = select_typiclust_points_dynamic(
            embeddings=train_embeddings,
            cluster_labels=cluster_labels,
            labeled_set=labeled_set_dynamic,
            budget=increment,
            use_hybrid=use_hybrid,
            k_neighbors=20
        )
        labeled_set_dynamic.update(new_points_dynamic)
        tpcrp_indices_dynamic = list(labeled_set_dynamic)
        acc_dynamic = resnet_eval(tpcrp_indices_dynamic)
        past_eval_accuracies.append(acc_dynamic)
        tpcrp_mean_dynamic.append(acc_dynamic)

        # Base TypiClust selection
        new_points_base = select_typiclust_points(
            embeddings=train_embeddings,
            cluster_labels=cluster_labels,
            labeled_set=labeled_set_base,
            budget=increment,
            k_neighbors=20
        )
        labeled_set_base.update(new_points_base)
        tpcrp_indices_base = list(labeled_set_base)
        acc_base = resnet_eval(tpcrp_indices_base)
        tpcrp_mean_base.append(acc_base)

        previous_budget = budget

        print(f"TypiClust Dynamic: {acc_dynamic:.4f}, TypiClust Base: {acc_base:.4f}")

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(budgets, tpcrp_mean_dynamic, marker='s', label='TypiClust Switch')
    plt.plot(budgets, tpcrp_mean_base, marker='^', label='TypiClust Base')
    plt.xlabel("Budget")
    plt.ylabel("Test Accuracy")
    plt.title("ResNet Accuracy vs Budget")
    plt.legend()
    plt.grid(True)
    plt.savefig("resnet_accuracy_vs_budget.pdf")
    print("saved resnet_accuracy_vs_budget.pdf")
    plt.show()