import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import random
import numpy as np

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


# Load MUTAG dataset
dataset = TUDataset(root="C:\\Users\\bini\\Desktop\\DeepTGNN", name='MUTAG')

# Check if GPU is available
if torch.cuda.is_available():
    # If GPU is available, use it; otherwise, use CPU
    DEVICE = torch.device("cuda")
    print("GPU is available.")
else:
    DEVICE = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# Define the sizes for train, val, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Print the number of graphs in each set
print(f"Number of graphs in the training set: {len(train_dataset)}")
print(f"Number of graphs in the validation set: {len(val_dataset)}")
print(f"Number of graphs in the test set: {len(test_dataset)}")

# Move the data loaders to the selected device
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)

class GraphThinkingGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_projection_layers=1, num_recurrent_layers=4, num_output_layers=2):
        super(GraphThinkingGNN, self).__init__()
        
        # Projection layer (p)
        self.projection = self.make_layers(in_channels, hidden_channels, num_projection_layers)
        
        # Recurrent block (r)
        self.recurrent_block = self.make_layers(hidden_channels, hidden_channels, num_recurrent_layers)
        
        # Output head (h)
        self.output_head = self.make_layers(hidden_channels + in_channels, out_channels, num_output_layers)
    
    def make_layers(self, in_channels, out_channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(GCNConv(in_channels, out_channels))
            # You can add other layers if needed, such as BatchNorm, Dropout, etc.
            in_channels = out_channels  # Update in_channels for the next layer
        return nn.Sequential(*layers)
    
    def recurrent_block_iterations(self, x, orig_feats, edge_index, num_iterations):
        for _ in range(num_iterations):
            for layer in self.recurrent_block:
                x = layer(x, edge_index)
                x = F.relu(x)  # Add ReLU here
        x = torch.cat([x, orig_feats], dim=1)
        return x
    
    def forward(self, data, is_training=True):
        orig_feats = data.x
        x, edge_index = data.x, data.edge_index
        
        # Projection layer (p)
        for layer in self.projection:
            x = layer(x, edge_index)
            x = F.relu(x)  # Add ReLU here
            x = nn.Dropout(p=0.4)(x)
        
        # Training recurrence (with concatenation)
        if is_training:
            num_iterations = 1
            x = self.recurrent_block_iterations(x, orig_feats, edge_index, num_iterations)
            x = F.relu(x)  # Add ReLU here
            x = nn.Dropout(p=0.4)(x)
        
        # Testing recurrence (no concatenation)
        if not is_training:
            test_iterations = 3
            x = self.recurrent_block_iterations(x, orig_feats, edge_index, test_iterations)
            x = F.relu(x)  # Add ReLU here
            x = nn.Dropout(p=0.4)(x)
        
        # Output head (h)
        for layer in self.output_head:
            x = layer(x, edge_index)
            x = F.relu(x)  # Add ReLU here
            x = nn.Dropout(p=0.4)(x)
        
        # Global pooling for aggregation
        x = global_mean_pool(x, data.batch)
        return x

# Initialize the model
model = GraphThinkingGNN(in_channels=dataset.num_node_features, hidden_channels=128, out_channels=2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 100
detach_interval = 75  # Adjust as needed
# early_stop_patience = 10
# best_val_accuracy = 0.0
# counter = 0  # Counter for the number of epochs with no improvement

if __name__ == '__main__':
    # Training loop
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

