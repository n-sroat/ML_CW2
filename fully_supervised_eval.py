import numpy as np
import torch
import matplotlib.pyplot as plt

from step2 import generate_cluster_labels
from typiclust import select_typiclust_points
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

import torch.nn as nn
import torch.optim as optim
from sampling import k_center_greedy


def resnet_eval(
    selected_indices,
    num_classes=10,
    num_epochs=100,
    lr=0.025,
    momentum=0.9,
    batch_size=128,
    use_random=False,
    random_seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu"
):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    if use_random:
        torch.manual_seed(random_seed)
        selected_indices = torch.randperm(len(train_dataset))[:len(selected_indices)].tolist()

    train_subset = Subset(train_dataset, selected_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=min(batch_size, len(train_subset)),
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )

    model = models.resnet18(num_classes=num_classes)

    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        nesterov=True,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    model.train()

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


if __name__ == "__main__":
    # Load embeddings
    train_embeddings_base = np.load("cifar10_embeddings_train.npy")
    train_embeddings = torch.tensor(train_embeddings_base, dtype=torch.float32)

    # Budgets
    budgets = [10, 20, 30, 40, 50, 60]
    num_repeats = 5

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

        # Evaluate TypiClust with full ResNet
        tpcrp_acc = resnet_eval(tpcrp_indices)
        tpcrp_mean_means.append(tpcrp_acc)
        tpcrp_mean_stds.append(0)  # deterministic

        random_accuracies = []
        for seed in range(num_repeats):
            acc = resnet_eval(
                tpcrp_indices,  # same size
                use_random=True,
                random_seed=seed
            )
            random_accuracies.append(acc)
        random_means.append(np.mean(random_accuracies))
        random_stds.append(np.std(random_accuracies))

        coreset_accuracies = []
        for seed in range(num_repeats):
            np.random.seed(seed)
            coreset_indices = k_center_greedy(train_embeddings_base, budget)
            acc = resnet_eval(coreset_indices)
            coreset_accuracies.append(acc)
        coreset_means.append(np.mean(coreset_accuracies))
        coreset_stds.append(np.std(coreset_accuracies))

        # Print summary
        print(f"Random: {random_means[-1]:.4f} ± {random_stds[-1]:.4f}")
        print(f"Coreset: {coreset_means[-1]:.4f} ± {coreset_stds[-1]:.4f}")
        print(f"TypiClust: {tpcrp_acc:.4f}")

    plt.figure(figsize=(8, 6))

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

    plt.plot(
        budgets,
        tpcrp_mean_means,
        marker='s',
        label='TypiClust'
    )

    plt.xlabel("Budget")
    plt.ylabel("Test Accuracy")
    plt.title("ResNet Accuracy vs Budget")
    plt.legend()
    plt.grid(True)

    plt.savefig("resnet_accuracy_vs_budget.pdf")
    print("Figure saved as resnet_accuracy_vs_budget.pdf")
    plt.show()