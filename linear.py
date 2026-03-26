import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Union, Tuple

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