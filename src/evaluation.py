import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs["logits"]
            preds = (torch.sigmoid(logits) > 0.5).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return {"accuracy": acc, "f1": f1}