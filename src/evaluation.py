import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef


def evaluate(model, dataloader, device, compute_loss: bool = True):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs_for_auc = [] 
    total_loss = 0

    with torch.inference_mode():
        for batch in dataloader:
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
            }
            if compute_loss: # Only add labels if we need to compute loss
                model_inputs['labels'] = labels
            
            if lengths is not None:
                model_inputs['lengths'] = lengths
            
            # Apply torch.autocast for MPS and CUDA
            if device.type == 'mps':
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    outputs = model(**model_inputs)
            elif device.type == 'cuda':
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**model_inputs)
            else: # CPU
                outputs = model(**model_inputs)

            if compute_loss:
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                else:
                    # This case should ideally not happen if labels were passed and model is in training mode or loss is expected
                    # For eval, if compute_loss is True, we expect loss. If model doesn't return it, it's an issue.
                    print("Warning: compute_loss was True, but model did not return loss.")
            
            logits = outputs.logits

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
        'loss': avg_loss if compute_loss and len(dataloader) > 0 else None, # Return None if loss was not computed
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'mcc': mcc
    }