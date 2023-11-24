import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphThinkingGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_projection_layers, num_recurrent_layers, num_output_layers):
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
        
        # Training recurrence (with recall concatenation)
        if is_training:
            train_iterations = 1
            x = self.recurrent_block_iterations(x, orig_feats, edge_index, train_iterations)
            x = F.relu(x)  # Add ReLU here
            x = nn.Dropout(p=0.4)(x)
        
        # Testing recurrence (with recall concatenation)
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


