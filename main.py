import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from data.dataloader import create_data_loaders
from model.thinking_gnn import GraphThinkingGNN
import numpy as np
import random

"""
This script is just the beginning of recall-GNN  for graph classification
using the MUTAG dataset. It could be a starting point.

The model architecture, GraphThinkingGNN, consists of three main parts:
1. Projection Layer (p): Transforms input node features using Graph Convolutional Networks (GCN).
2. Recurrent Block (r): Applies a stack of GCN layers for capturing graph context over multiple iterations.
3. Output Head (h): Produces the final graph-level classification using GCN layers.

The idea could be to train the model on a small set of graphs (in this case, MUTAG dataset) and evaluate its
generalization performance on a larger set (here I randomly split 30/20/50). This concept can be extended to node classification tasks on real-world
datasets, exploring the GNN's ability to generalize.

I haven't tuned any hyperparameters yet.
Feel free to change whatever needs to be changed.
"""

SEED=71
# Set random seed 
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("GPU is available.")
else:
    DEVICE = torch.device("cpu")
    print("GPU is not available. Using CPU.")

def main():
    # prompting number of layers
    parser = argparse.ArgumentParser(description='GraphThinkingGNN')
    parser.add_argument('--num_projection_layers', type=int, default=1, help='Number of projection layers')
    parser.add_argument('--num_recurrent_layers', type=int, default=4, help='Number of recurrent layers')
    parser.add_argument('--num_output_layers', type=int, default=2, help='Number of output layers')
    parser.add_argument('--train_iterations', type=int, default=1, help='Number of iterations for training recurrence')
    parser.add_argument('--test_iterations', type=int, default=3, help='Number of iterations for testing recurrence')

    args = parser.parse_args()

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders()

    # Initialize model
    model = GraphThinkingGNN(
        in_channels=train_loader.dataset[0].num_node_features,
        hidden_channels=128,
        out_channels=2,
        num_projection_layers=args.num_projection_layers,
        num_recurrent_layers=args.num_recurrent_layers,
        num_output_layers=args.num_output_layers
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 100
    detach_interval = 75  

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch, num_iterations=args.train_iterations, is_training=True)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()

            # Detach node features periodically
            if epoch % detach_interval == 0:
                batch.x.detach_()

    # Evaluation
    model.eval()  
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            output = model(batch, num_iterations=args.test_iterations, is_training=False)

            # Get the predicted labels
            _, predicted_labels = torch.max(output, 1)
            correct_predictions += (predicted_labels == batch.y).sum().item()
            total_samples += batch.y.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
