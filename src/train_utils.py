import math
from torch import nn
import torch
import torch.nn.functional as F


class SentimentWeightedLoss(nn.Module):
    """BCEWithLogits + dynamic weighting.

    We weight each sample by:
      • length_weight:  sqrt(num_tokens) / sqrt(max_tokens)
      • confidence_weight: |sigmoid(logits) - 0.5|  (higher confidence ⇒ larger weight)

    The two weights are combined multiplicatively then normalized.
    """

    def __init__(self):
        super().__init__()
        # Initialize BCE loss without reduction, since we're applying per-sample weights
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.min_len_weight_sqrt = 0.1  # Minimum length weight

    def forward(self, logits, targets, lengths):
        base_loss = self.bce(logits.view(-1), targets.float())  # shape [B]
    
        prob = torch.sigmoid(logits.view(-1))
        confidence_weight = (prob - 0.5).abs() * 2  # ∈ [0,1]

        if lengths.numel() == 0:
            # Handle empty batch: return 0.0 loss or mean of base_loss if it's also empty (becomes nan then)
            # If base_loss on empty input is empty tensor, mean is nan. So return 0.0 is safer.
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)
        
        length_weight = torch.sqrt(lengths.float()) / math.sqrt(lengths.max().item())
        length_weight = length_weight.clamp(self.min_len_weight_sqrt, 1.0) # Clamp to avoid extreme weights

        weights = confidence_weight * length_weight
        weights = weights / (weights.mean() + 1e-8)  # normalize so E[w]=1
        return (base_loss * weights).mean()




class SentimentFocalLoss(nn.Module):
    """
    This probably overcomplicated loss function incorporates:
    1. Base BCEWithLogitsLoss.
    2. Label Smoothing.
    3. Focal Loss modulation to focus more on hard examples (can be reversed to focus on easy examples).
    4. Sample weighting based on review length.
    5. Sample weighting based on prediction confidence.

    The final loss for each sample is calculated roughly as:
    Loss_sample = FocalModulator(pt, gamma) * BCE(logits, smoothed_targets) * NormalizedExternalWeight
    NormalizedExternalWeight = (ConfidenceWeight * LengthWeight) / Mean(ConfidenceWeight * LengthWeight)
    """

    def __init__(self, gamma_focal: float = 0.1, label_smoothing_epsilon: float = 0.05):
        """
        Args:
            gamma_focal (float): Gamma parameter for Focal Loss.
                - If gamma_focal > 0 (e.g., 2.0), applies standard Focal Loss,
                  down-weighting easy examples (focus on hard examples).
                - If gamma_focal < 0 (e.g., -2.0), applies a reversed Focal Loss,
                  down-weighting hard examples (focus on easy examples by up-weighting pt).
                - If gamma_focal = 0, no Focal Loss modulation is applied.
            label_smoothing_epsilon (float): Epsilon for label smoothing. (0.0 <= epsilon < 1.0)
                - If 0.0, no label smoothing is applied. Converts hard labels (0, 1)
                  to soft labels (epsilon, 1-epsilon).
        """
        super().__init__()
        if not (0.0 <= label_smoothing_epsilon < 1.0):
            raise ValueError("label_smoothing_epsilon must be between 0.0 and <1.0.")
        
        self.gamma_focal = gamma_focal
        self.label_smoothing_epsilon = label_smoothing_epsilon
        # Initialize BCE loss without reduction, since we're applying per-sample weights
        self.bce_loss_no_reduction = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Computes the custom loss.

        Args:
            logits (torch.Tensor): Raw logits from the model. Expected shape [B] or [B, 1].
            targets (torch.Tensor): Ground truth labels (0 or 1). Expected shape [B] or [B, 1].
            lengths (torch.Tensor): Number of tokens in each review. Expected shape [B].

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        B = logits.size(0)
        if B == 0: # Handle empty batch case
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_flat = logits.view(-1)
        original_targets_flat = targets.view(-1).float() # Ensure targets are float

        # 1. Label Smoothing
        if self.label_smoothing_epsilon > 0:
            # Smooth 1 to (1 - epsilon), and 0 to epsilon
            targets_for_bce = original_targets_flat * (1.0 - self.label_smoothing_epsilon) + \
                              (1.0 - original_targets_flat) * self.label_smoothing_epsilon
        else:
            targets_for_bce = original_targets_flat

        # 2. Calculate Base BCE loss terms (using potentially smoothed targets)
        base_bce_loss_terms = self.bce_loss_no_reduction(logits_flat, targets_for_bce)

        # 3. Focal Loss Modulation Component
        # For the focal modulator, 'pt' is the probability assigned by the model to the *original* ground truth class.
        probs = torch.sigmoid(logits_flat)
        # pt: probability of the original true class
        pt = torch.where(original_targets_flat.bool(), probs, 1.0 - probs)

        focal_modulator = torch.ones_like(pt) # Default to 1 (no modulation if gamma_focal is 0)
        if self.gamma_focal > 0:  # Standard Focal Loss: (1-pt)^gamma. Focus on hard examples (pt is small).
            focal_modulator = (1.0 - pt + 1e-8).pow(self.gamma_focal) # Epsilon for stability if pt is 1
        elif self.gamma_focal < 0:  # Reversed Focal: (pt)^|gamma|. Focus on easy examples (pt is large).
            focal_modulator = (pt + 1e-8).pow(abs(self.gamma_focal)) # Epsilon for stability if pt is 0
        
        modulated_loss_terms = focal_modulator * base_bce_loss_terms

        # 4. Confidence Weighting (based on how far probability is from 0.5)
        # Uses the same `probs` calculated for focal `pt`.
        confidence_w = (probs - 0.5).abs() * 2.0  # Scales to range [0, 1]

        # 5. Length Weighting (longer reviews potentially weighted more)
        lengths_flat = lengths.view(-1).float()
        max_len_in_batch = lengths_flat.max().item()
        
        if max_len_in_batch == 0: # Edge case: if all reviews in batch have 0 length
            length_w = torch.ones_like(lengths_flat)
        else:
            # Normalize by sqrt of max length in the current batch. Add epsilon for stability.
            length_w = torch.sqrt(lengths_flat) / (math.sqrt(max_len_in_batch) + 1e-8)
            length_w = torch.clamp(length_w, 0.0, 1.0) # Ensure weights are capped at 1

        # 6. Combine External Weights (Confidence and Length)
        # These weights are applied ON TOP of the focal-modulated loss terms.
        external_weights = confidence_w * length_w
        
        # Normalize these combined external_weights so their mean is approximately 1.
        # This prevents the weighting scheme from drastically changing the overall loss magnitude.
        if external_weights.sum() > 1e-8: # Avoid division by zero if all weights are zero
             normalized_external_weights = external_weights / (external_weights.mean() + 1e-8)
        else: # If all external weights are zero, use ones to not nullify the loss.
             normalized_external_weights = torch.ones_like(external_weights)

        # 7. Apply Normalized External Weights to the (Focal) Modulated Loss Terms
        final_loss_terms_per_sample = modulated_loss_terms * normalized_external_weights
        
        # 8. Final Reduction: Mean of the per-sample losses
        loss = final_loss_terms_per_sample.mean()
        
        return loss
