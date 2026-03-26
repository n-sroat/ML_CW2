"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
Windows-safe SimCLR training script
"""
import argparse
import os
import torch
import numpy as np
import gc
from termcolor import colored

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset, \
    get_val_dataset, get_train_dataloader, get_val_dataloader, \
    get_train_transformations, get_val_transformations, get_optimizer, \
    adjust_learning_rate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from utils.collate import collate_custom


# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env', help='Config file for the environment')
parser.add_argument('--config_exp', help='Config file for the experiment')
args = parser.parse_args()


def main():
    # Load config
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p).cuda()
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)

    # CuDNN benchmark
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations(p)

    print('Train transforms:', train_transforms)
    print('Validation transforms:', val_transforms)

    train_dataset = get_train_dataset(
        p,
        train_transforms,
        to_augmented_dataset=True,
        split='train+unlabeled'
    )

    val_dataset = get_val_dataset(p, val_transforms)

    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)

    print('Dataset contains {}/{} train/val samples'.format(
        len(train_dataset),
        len(val_dataset))
    )

    # Memory banks only needed after training
    print(colored('Build MemoryBank', 'blue'))

    base_dataset = get_train_dataset(p, val_transforms, split='train')
    base_dataloader = get_val_dataloader(p, base_dataset)

    memory_bank_base = MemoryBank(
        len(base_dataset),
        p['model_kwargs']['features_dim'],
        p['num_classes'],
        p['criterion_kwargs']['temperature']
    )
    memory_bank_base.cpu()

    memory_bank_val = MemoryBank(
        len(val_dataset),
        p['model_kwargs']['features_dim'],
        p['num_classes'],
        p['criterion_kwargs']['temperature']
    )
    memory_bank_val.cpu()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p).cuda()
    print('Criterion is {}'.format(criterion.__class__.__name__))

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    start_epoch = 0

    if os.path.exists(p['pretext_checkpoint']):
        print(colored(f'Restart from checkpoint {p["pretext_checkpoint"]}', 'blue'))

        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')

        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])

        model.cuda()

        start_epoch = checkpoint['epoch']

    else:
        print(colored(f'No checkpoint file at {p["pretext_checkpoint"]}', 'blue'))

    # Training loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):

        print(colored(f'Epoch {epoch}/{p["epochs"]}', 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust learning rate
        lr = adjust_learning_rate(p, optimizer, epoch)
        print(f'Adjusted learning rate to {lr:.5f}')

        # Train
        print('Train ...')
        simclr_train(train_dataloader, model, criterion, optimizer, epoch)

        torch.cuda.empty_cache()
        gc.collect()

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == p['epochs'] - 1:
            print('Checkpoint ...')
            torch.save({
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'epoch': epoch + 1
            }, p['pretext_checkpoint'])

    # Save final model
    torch.save(model.state_dict(), p['pretext_model'])

    # Mine nearest neighbors (train)
    print(colored('Mine nearest neighbors (train)', 'blue'))
    fill_memory_bank(base_dataloader, model, memory_bank_base)

    topk = 20
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)

    print(f'Accuracy of top-{topk} nearest neighbors on train set is {100*acc:.2f}')

    np.save(p['topk_neighbors_train_path'], indices)

    # Mine nearest neighbors (val)
    print(colored('Mine nearest neighbors (val)', 'blue'))
    fill_memory_bank(val_dataloader, model, memory_bank_val)

    topk = 5
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)

    print(f'Accuracy of top-{topk} nearest neighbors on val set is {100*acc:.2f}')

    np.save(p['topk_neighbors_val_path'], indices)


if __name__ == '__main__':
    main()