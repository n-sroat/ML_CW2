import numpy as np
import torch
import matplotlib.pyplot as plt

from step2 import generate_cluster_labels
from typiclust import select_typiclust_points
from typiclust_modification import select_typiclust_points_dynamic, decide_strategy
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

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

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    if use_random:
        torch.manual_seed(random_seed)
        selected_indices = torch.randperm(len(train_dataset))[:len(selected_indices)].tolist()

    train_subset = Subset(train_dataset, selected_indices)
    train_loader = DataLoader(train_subset, batch_size=min(batch_size, len(train_subset)), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


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