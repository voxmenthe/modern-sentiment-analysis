import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'lengths'}
            # Cast labels to float for internal MSE loss calculation (when num_labels=1)
            if 'labels' in batch:
                batch['labels'] = batch['labels'].float()

            outputs = model(**batch)
            logits = outputs.logits # Use attribute access
            preds = (torch.sigmoid(logits) > 0.5).long().cpu()
            all_preds.extend(preds.tolist())
            # Ensure original labels (int/long) are used for metric calculation
            # Re-fetch or cast back if necessary, though cpu().tolist() should handle it.
            # Let's keep using batch['labels'] assuming it was moved to device but not necessarily cast back
            original_labels = batch["labels"].long().cpu().tolist() # Make sure we compare int labels
            all_labels.extend(original_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return {"accuracy": acc, "f1": f1}