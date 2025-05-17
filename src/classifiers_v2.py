import torch
import torch.nn as nn
import torch.nn.functional as F

class GranularMoELayer(nn.Module):
    """
    A Mixture-of-Experts layer with linear top-k routing and ReLU experts.
    Note: This is a simplified implementation focusing on the layer structure,
    not a complete model.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_experts: int,
                 num_active_experts: int, # This is the 'granularity'
                 expert_hidden_size: int):
        """
        Args:
            input_size: Dimension of the input tensor.
            output_size: Dimension of the output tensor.
            num_experts: Total number of experts in the layer.
            num_active_experts: Number of experts activated per input (k).
            expert_hidden_size: Hidden size for each expert's feed-forward network.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.expert_hidden_size = expert_hidden_size

        if num_active_experts > num_experts:
            raise ValueError("Number of active experts cannot exceed total experts.")

        # Gating network (linear router as analyzed in the paper)
        # Output size is num_experts for scores for each expert
        self.gate = nn.Linear(input_size, num_experts)

        # Expert networks (2-layer feed-forward with ReLU)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, expert_hidden_size),
                nn.ReLU(),
                nn.Linear(expert_hidden_size, output_size)
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE layer.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        batch_size = x.shape[0]

        # 1. Compute expert scores using the gating network
        # scores shape: (batch_size, num_experts)
        gate_scores = self.gate(x)

        # 2. Select top-k experts based on scores (Top-k routing)
        # The paper analyzes linear routing (top-k inner products),
        # which corresponds to selecting based on these scores.
        # topk_scores shape: (batch_size, num_active_experts)
        # topk_indices shape: (batch_size, num_active_experts)
        topk_scores, topk_indices = torch.topk(gate_scores, self.num_active_experts, dim=-1)

        # 3. Create a mask for activated experts
        # activated_experts_mask shape: (batch_size, num_experts)
        # This mask will be 1 for selected experts, 0 otherwise.
        activated_experts_mask = torch.zeros_like(gate_scores, dtype=x.dtype)
        activated_experts_mask.scatter_(-1, topk_indices, 1.0) # Scatter 1s at topk_indices

        # Optional: Normalize top-k scores if used for weighting.
        # The paper's primary theoretical result for constant activation
        # didn't use weighted sums (Remark 2.1 mentions this as a variation).
        # For linear/ReLU experts in the main theorems, the output is a sum
        # over experts in the activated set S (Eq 2.2), implying unweighted sum.
        # We will follow the unweighted sum approach from Eq 2.2.
        # If weighting were desired, we could do:
        # weights = F.softmax(topk_scores, dim=-1)
        # The scatter operation above already provides a binary mask for selection.

        # 4. Execute the forward pass for each expert
        # We can optimize this by only processing inputs for the batches
        # where an expert is activated, but for simplicity and clarity here,
        # we'll process all experts and use the mask later.
        # A more efficient implementation would reroute inputs.

        # expert_outputs shape: (batch_size, num_experts, output_size)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # 5. Combine outputs from activated experts
        # Expand the mask and expert outputs to match for element-wise multiplication
        # mask_expanded shape: (batch_size, num_experts, 1)
        mask_expanded = activated_experts_mask.unsqueeze(-1)

        # weighted_outputs shape: (batch_size, num_experts, output_size)
        # This effectively zeros out outputs from non-activated experts
        weighted_outputs = expert_outputs * mask_expanded

        # Sum across the expert dimension to get the final output for the batch
        # final_output shape: (batch_size, output_size)
        final_output = torch.sum(weighted_outputs, dim=1)

        # Note: The paper's Eq 2.2 suggests f(x) = sum_{j in S} M_j x for linear experts.
        # M_j is implicitly (A_j B_j) in their construction (Lemma B.9).
        # The `nn.Sequential` structure above implements A_j(sigma(B_j x)) which is
        # A_j ReLU(B_j x) for ReLU activation. The sum in Eq 2.2 is then implemented
        # by the torch.sum(weighted_outputs, dim=1) here.

        return final_output

# Example Usage (optional, keep commented out in the file itself)
# if __name__ == "__main__":
#     input_dim = 128
#     output_dim = 256
#     total_experts = 64 # num_experts
#     active_experts = 8 # granularity (k) - example of higher granularity
#     expert_hidden = 512
#
#     # Instantiate the layer
#     moe_layer = GranularMoELayer(input_dim, output_dim, total_experts, active_experts, expert_hidden)
#
#     # Create a dummy input batch
#     batch_size = 32
#     dummy_input = torch.randn(batch_size, input_dim)
#
#     # Pass through the layer
#     output = moe_layer(dummy_input)
#
#     print(f"Input shape: {dummy_input.shape}")
#     print(f"Output shape: {output.shape}")
#
#     # Verify that only k experts contribute to the output for each input in the batch
#     # This is implicitly handled by the scatter operation in the mask.
#     # To verify active experts per batch item, you'd inspect topk_indices.
#     # The sum of the mask entries per row should equal num_active_experts
#     # gate_scores = moe_layer.gate(dummy_input)
#     # topk_scores, topk_indices = torch.topk(gate_scores, moe_layer.num_active_experts, dim=-1)
#     # activated_mask = torch.zeros_like(gate_scores, dtype=dummy_input.dtype)
#     # activated_mask.scatter_(-1, topk_indices, 1.0)
#     # print(f"Sum of activated mask per batch item: {torch.sum(activated_mask, dim=-1)}")


class MoEClassifierHead(nn.Module):
    """
    A classifier head that uses a configurable number of GranularMoELayers.
    Each MoE block consists of LayerNorm, GranularMoELayer, and Dropout,
    with a skip connection. The GranularMoELayer uses ReLU activation
    internally for its experts.
    The number of MoE blocks defaults to 2.
    """
    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float,
                 num_experts: int, num_active_experts: int, expert_hidden_size: int,
                 num_hidden_layers: int = 2): # Configurable number of layers
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.moe_blocks = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.moe_blocks.append(
                nn.ModuleDict({
                    'norm': nn.LayerNorm(hidden_size),
                    'moe': GranularMoELayer(
                        input_size=hidden_size,
                        output_size=hidden_size, # Maintain dimensionality for skip and stacking
                        num_experts=num_experts,
                        num_active_experts=num_active_experts,
                        expert_hidden_size=expert_hidden_size
                    ),
                    # GranularMoELayer uses ReLU activation within its experts.
                    'dropout': nn.Dropout(dropout_prob)
                })
            )
        
        # Output Layer (same as original ClassifierHead)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features

        for block in self.moe_blocks:
            identity = x
            
            processed_x = block['norm'](x)
            moe_output = block['moe'](processed_x)
            # moe_output incorporates activation (ReLU) from the experts within GranularMoELayer
            dropped_output = block['dropout'](moe_output)
            x = dropped_output + identity  # Skip connection

        # Output Layer
        logits = self.out_proj(x)
        return logits