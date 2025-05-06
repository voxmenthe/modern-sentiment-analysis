import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            # Keep 'lengths' for the custom model's forward pass (required for loss calc)
            batch = {k: v.to(device) for k, v in batch.items()}
            # Remove casting labels to float, let the custom loss handle types
            # if 'labels' in batch:
            #     batch['labels'] = batch['labels'].float()

            outputs = model(**batch)
            logits = outputs.logits # Use attribute access
            preds = (torch.sigmoid(logits) > 0.5).long().cpu()
            all_preds.extend(preds.tolist())
            # Ensure original labels (int/long) are used for metric calculation
            original_labels = batch["labels"].long().cpu().tolist() # Make sure we compare int labels
            all_labels.extend(original_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return {"accuracy": acc, "f1": f1}