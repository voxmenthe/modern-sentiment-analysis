import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef


def evaluate(model, dataloader, device, *,
             compute_loss: bool = False, # Added: Ticket 3, used by Ticket 2
             max_batches: int | None = None): # Added: Ticket 2
    model.eval()
    all_preds = []
    all_labels = []
    all_probs_for_auc = [] 
    total_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader): # Added enumerate for max_batches logic
            if max_batches and i >= max_batches: # Added: Ticket 2
                break
            # Move batch to device, ensure all model inputs are covered
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch.get('lengths') # Get lengths from batch
            if lengths is None:
                # Fallback or error if lengths are expected but not found
                # For now, let's raise an error if using weighted loss that needs it
                # Or, if your model can run without it for some pooling strategies, handle accordingly
                # However, the error clearly states it's needed when labels are specified.
                pass # Or handle error: raise ValueError("'lengths' not found in batch, but required by model")
            else:
                lengths = lengths.to(device) # Move to device if found

            # Pass all necessary parts of the batch to the model
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                # 'labels': labels # Labels only added if compute_loss is True
            }

            if compute_loss: # Added: Ticket 3
                model_inputs['labels'] = batch['labels'].to(device)
                if lengths is not None: # Keep lengths if labels are present
                    model_inputs['lengths'] = lengths
            
            outputs = model(**model_inputs)
            
            # Loss calculation only if compute_loss is True
            if compute_loss: # Added: Ticket 3
                loss = outputs.loss
                total_loss += loss.item()
            else:
                loss = None # Ensure loss is None if not computed

            logits = outputs.logits

            # total_loss += loss.item() # Moved into compute_loss block
            
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

    avg_loss = (total_loss / len(dataloader)) if compute_loss and len(dataloader) > 0 else 0.0 # Modified: Ticket 3
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