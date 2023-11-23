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

SEED=71
# Set random seed for PyTorch
torch.manual_seed(SEED)

# Set random seed for CUDA (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Set random seed for NumPy
np.random.seed(SEED)

# Set random seed for Python built-in random module
random.seed(SEED)

# Check if GPU is available
if torch.cuda.is_available():
    # If GPU is available, use it; otherwise, use CPU
    DEVICE = torch.device("cuda")
    print("GPU is available.")
else:
    DEVICE = torch.device("cpu")
    print("GPU is not available. Using CPU.")


# main function
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
    detach_interval = 75  # Adjust as needed

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch, is_training=True)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()

            # Detach node features periodically
            if epoch % detach_interval == 0:
                batch.x.detach_()

    # Evaluation loop on the test set
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            output = model(batch, is_training=False)

            # Get the predicted labels
            _, predicted_labels = torch.max(output, 1)

            # Count the number of correct predictions
            correct_predictions += (predicted_labels == batch.y).sum().item()

            # Count the total number of samples
            total_samples += batch.y.size(0)

    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
