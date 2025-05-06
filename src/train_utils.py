import math
from torch import nn
import torch


class SentimentWeightedLoss(nn.Module):
    """BCEWithLogits + dynamic weighting.

    We weight each sample by:
      • length_weight:  sqrt(num_tokens) / sqrt(max_tokens)
      • confidence_weight: |sigmoid(logits) - 0.5|  (higher confidence ⇒ larger weight)

    The two weights are combined multiplicatively then normalized.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, lengths):
        base_loss = self.bce(logits.view(-1), targets.float())  # shape [B]

        prob = torch.sigmoid(logits.view(-1))
        confidence_weight = (prob - 0.5).abs() * 2  # ∈ [0,1]
        length_weight = torch.sqrt(lengths.float()) / math.sqrt(lengths.max().item())

        weights = confidence_weight * length_weight
        weights = weights / (weights.mean() + 1e-8)  # normalize so E[w]=1
        return (base_loss * weights).mean()