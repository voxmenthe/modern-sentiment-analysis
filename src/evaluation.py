import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
from torch.cuda.amp import autocast


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs_for_auc = [] 
    total_loss = 0

    use_cuda = device.type == 'cuda'

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device, ensure all model inputs are covered
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            lengths = batch.get('lengths') # Get lengths from batch
            if lengths is None:
                # Fallback or error if lengths are expected but not found
                # For now, let's raise an error if using weighted loss that needs it
                # Or, if your model can run without it for some pooling strategies, handle accordingly
                # However, the error clearly states it's needed when labels are specified.
                pass # Or handle error: raise ValueError("'lengths' not found in batch, but required by model")
            else:
                lengths = lengths.to(device, non_blocking=True) # Move to device if found

            # Pass all necessary parts of the batch to the model
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            if lengths is not None:
                model_inputs['lengths'] = lengths

            if use_cuda:
                with autocast():
                    outputs = model(**model_inputs)
            else:
                outputs = model(**model_inputs)
            
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