import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from data.dataloader import create_data_loaders
from model.thinking_gnn import GraphThinkingGNN

SEED = 71
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
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders()

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

