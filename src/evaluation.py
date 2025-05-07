import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs_for_auc = [] 
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            
            if logits.shape[1] > 1: 
                preds = torch.argmax(logits, dim=1)
            else: 
                preds = (torch.sigmoid(logits) > 0.5).long() 
            all_preds.extend(preds.cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy()) 

            if logits.shape[1] > 1: 
                probs = torch.softmax(logits, dim=1)[:, 1] 
                all_probs_for_auc.extend(probs.cpu().numpy())
            else: 
                probs = torch.sigmoid(logits) 
                all_probs_for_auc.extend(probs.squeeze().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    try:
        roc_auc = roc_auc_score(all_labels, all_probs_for_auc)
    except ValueError as e:
        print(f"Could not calculate AUC-ROC: {e}. Labels: {list(set(all_labels))[:10]}. Probs example: {all_probs_for_auc[:5]}. Setting to 0.0")
        roc_auc = 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'mcc': mcc
    }