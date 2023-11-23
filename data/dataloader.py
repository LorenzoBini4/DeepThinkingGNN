import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split

def create_data_loaders():
    current_directory = os.getcwd()
    dataset_name = 'MUTAG'
    dataset_path = os.path.join(current_directory, dataset_name)
    dataset = TUDataset(root=dataset_path, name=dataset_name)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print(f"Number of graphs in the training set: {len(train_dataset)}")
    print(f"Number of graphs in the validation set: {len(val_dataset)}")
    print(f"Number of graphs in the test set: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)

    return train_loader, val_loader, test_loader
