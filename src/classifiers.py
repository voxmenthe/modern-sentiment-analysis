from torch import nn
import torch


class EnhancedClassifierHead(nn.Module):
    """A 3-layer classifier head with GELU, LayerNorm, and Skip Connections."""
    def __init__(self, hidden_size, num_labels, dropout_prob):
        super().__init__()
        # Layer 1
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)

        # Layer 2
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)

        # Output Layer
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        # Layer 1 + Skip
        identity1 = features
        x = self.norm1(features)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = x + identity1 # Skip connection 1

        # Layer 2 + Skip
        identity2 = x
        x = self.norm2(x)
        x = self.dense2(x)
        x = self.activation(x) # Re-use activation
        x = self.dropout2(x)
        x = x + identity2 # Skip connection 2

        # Output Layer
        logits = self.out_proj(x)
        return logits


class AdvancedClassifierHead(nn.Module):
    """
    A classifier head using FFN-style expansion (input -> 4*hidden -> hidden -> labels).
    Takes concatenated CLS + Mean Pooled features as input.
    """
    def __init__(self, input_size, hidden_size, num_labels, dropout_prob):
        super().__init__()
        intermediate_size = hidden_size * 4 # FFN expansion factor

        # Layer 1 (Expansion)
        self.norm1 = nn.LayerNorm(input_size)
        self.dense1 = nn.Linear(input_size, intermediate_size)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)

        # Layer 2 (Projection back down)
        self.norm2 = nn.LayerNorm(intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        # Activation and Dropout are applied after projection
        self.dropout2 = nn.Dropout(dropout_prob)

        # Output Layer
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        # Layer 1
        x = self.norm1(features)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.norm2(x)
        x = self.dense2(x)
        x = self.activation(x) # Activation after projection
        x = self.dropout2(x)

        # Output Layer
        logits = self.out_proj(x)
        return logits


# --- NEW CLASS --- 
class ConcatEnhancedClassifierHead(nn.Module):
    """
    An enhanced classifier head designed for concatenated CLS + Mean Pooling input.
    Includes an initial projection layer before the standard enhanced block.
    """
    def __init__(self, input_size, hidden_size, num_labels, dropout_prob):
        super().__init__()
        # Initial projection from concatenated size (2*hidden) down to hidden_size
        self.initial_projection = nn.Linear(input_size, hidden_size)
        self.initial_norm = nn.LayerNorm(hidden_size) # Norm after projection
        self.initial_activation = nn.GELU()
        self.initial_dropout = nn.Dropout(dropout_prob)

        # --- Start of Enhanced Block (operating on hidden_size) ---
        # Layer 1
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU() # Can re-use activation instance
        self.dropout1 = nn.Dropout(dropout_prob)

        # Layer 2
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)

        # Output Layer
        self.out_proj = nn.Linear(hidden_size, num_labels)
        # --- End of Enhanced Block ---

    def forward(self, features):
        # Initial Projection Step
        x = self.initial_projection(features)
        x = self.initial_norm(x)
        x = self.initial_activation(x)
        x = self.initial_dropout(x)
        # x is now shape (batch_size, hidden_size)

        # --- Start Enhanced Block ---
        # Layer 1 + Skip
        identity1 = x # Skip connection starts after initial projection
        x_res = self.norm1(x)
        x_res = self.dense1(x_res)
        x_res = self.activation(x_res)
        x_res = self.dropout1(x_res)
        x = x + x_res # Add skip connection

        # Layer 2 + Skip
        identity2 = x
        x_res = self.norm2(x)
        x_res = self.dense2(x_res)
        x_res = self.activation(x_res)
        x_res = self.dropout2(x_res)
        x = x + x_res # Add skip connection
        # --- End Enhanced Block ---

        # Output Layer
        logits = self.out_proj(x)
        return logits
# --- END NEW CLASS ---
