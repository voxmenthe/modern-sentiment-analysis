import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
from torch.amp import autocast


def evaluate(model, dataloader, device, compute_loss=True):
    """
    Evaluate model on the provided dataloader.

    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        compute_loss: Whether to compute loss (can be skipped for faster evaluation)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs_for_auc = []
    total_loss = 0.0

    use_cuda = device.type == 'cuda'

    # Use inference_mode instead of no_grad for better performance
    with torch.inference_mode():
        for batch in dataloader:
            # Move entire batch to device at once with non_blocking=True
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Skip loss computation if not needed (faster evaluation)
            if not compute_loss and 'labels' in batch:
                labels_backup = batch['labels'].clone()  # Save for metrics
                model_inputs = {k: v for k, v in batch.items() if k != 'labels'}

                if use_cuda:
                    with autocast('cuda'):
                        outputs = model(**model_inputs)
                else:
                    outputs = model(**model_inputs)

                labels = labels_backup
            else:
                # Regular forward pass with loss computation
                if use_cuda:
                    with autocast('cuda'):
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)

                if compute_loss and hasattr(outputs, 'loss'):
                    # Avoid unnecessary CPU-GPU sync with item()
                    total_loss += outputs.loss.detach()

                labels = batch['labels']

            logits = outputs.logits

            # Compute predictions efficiently
            if logits.shape[1] > 1:
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                preds = (torch.sigmoid(logits) > 0.5).long()
                probs = torch.sigmoid(logits).squeeze()

            # Collect results - move to CPU in batches
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs_for_auc.append(probs.cpu())

    # Concatenate all tensors before moving to numpy for better efficiency
    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_probs_tensor = torch.cat(all_probs_for_auc, dim=0)

    # Convert to numpy arrays once
    all_preds_np = all_preds_tensor.numpy()
    all_labels_np = all_labels_tensor.numpy()
    all_probs_np = all_probs_tensor.numpy()

    # Calculate metrics
    accuracy = accuracy_score(all_labels_np, all_preds_np)
    f1 = f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    precision = precision_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    recall = recall_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(all_labels_np, all_preds_np)

    # Calculate AUC-ROC
    try:
        roc_auc = roc_auc_score(all_labels_np, all_probs_np)
    except ValueError as e:
        print(f"Could not calculate AUC-ROC: {e}. Setting to 0.0")
        roc_auc = 0.0

    # Calculate loss if computed
    if compute_loss:
        if isinstance(total_loss, torch.Tensor):
            avg_loss = (total_loss / len(dataloader)).item()
        else:
            avg_loss = total_loss / len(dataloader)
    else:
        avg_loss = None

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'mcc': mcc
    }