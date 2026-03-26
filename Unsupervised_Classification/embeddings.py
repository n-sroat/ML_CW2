import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.cifar import CIFAR10
import numpy as np
from models.resnet_cifar import resnet18
import os
from pathlib import Path

batch_size = 512
num_workers = 0

model_path = Path('./results/cifar-10/pretext/model.pth.tar')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

train_split = False  # Set True for train embeddings, False for test embeddings
dataset = CIFAR10(
    root=Path('./dataset'),
    train=train_split,
    transform=transform,
    download=True
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,  # keep indices aligned with dataset.targets
    num_workers=num_workers
)

backbone_dict = resnet18()
model = backbone_dict['backbone']

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

# Remove projection head weights if present
clean_state_dict = {k.replace('backbone.', ''): v
                    for k, v in state_dict.items() if k.startswith('backbone.')}
model.load_state_dict(clean_state_dict, strict=False)

model.to(device)
model.eval()

embeddings = []
labels = []

with torch.no_grad():
    for batch in dataloader:
        images = batch['image'].to(device)
        targets = batch['target']  # aligned labels

        emb = model(images)
        emb = F.normalize(emb, dim=1)

        embeddings.append(emb.cpu().numpy())
        labels.append(targets.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)

suffix = "train" if train_split else "test"
np.save(f'cifar10_embeddings{suffix}.npy', embeddings)
np.save(f'cifar10_labels{suffix}.npy', labels)

print(f"Saved embeddings ({suffix}):", embeddings.shape)
print(f"Saved labels ({suffix}):", labels.shape)