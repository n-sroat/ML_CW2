from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torch

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
