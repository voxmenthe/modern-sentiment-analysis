import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
from models import ModernBertForSentiment # Assuming models.py is in the same directory
from tqdm import tqdm # Add this import for the progress bar


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs_for_auc = [] 
    total_loss = 0
    num_batches = len(dataloader)
    processed_batches = 0

    with torch.no_grad():
        for batch in dataloader: # dataloader here should not be pre-wrapped with tqdm by the caller if we yield progress
            processed_batches += 1
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
                'labels': labels
            }
            if lengths is not None:
                model_inputs['lengths'] = lengths
            
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

            # Populate probabilities for AUC calculation
            if logits.shape[1] > 1: 
                # Multi-class or multi-label, assuming positive class is at index 1 for binary-like AUC
                probs_for_auc = torch.softmax(logits, dim=1)[:, 1] 
            else: 
                # Binary classification with a single logit output
                probs_for_auc = torch.sigmoid(logits).squeeze() 
            all_probs_for_auc.extend(probs_for_auc.cpu().numpy())

            # Yield progress update
            progress_update_frequency = max(1, num_batches // 20) # Ensure at least 1 to avoid modulo zero
            if processed_batches % progress_update_frequency == 0 or processed_batches == num_batches: # Update roughly 20 times + final
                yield f"Processed {processed_batches}/{num_batches} batches ({processed_batches/num_batches*100:.2f}%)"
            
    avg_loss = total_loss / num_batches
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

    results = {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'average_loss': avg_loss
    }
    yield f"Processed {processed_batches}/{num_batches} batches (100.00%)" # Ensure final progress update
    yield "Evaluation complete. Compiling results..."
    yield results

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from inference import SentimentInference # Assuming inference.py is in the same directory
    import yaml
    from transformers import AutoTokenizer, AutoConfig
    from models import ModernBertForSentiment # Assuming models.py is in the same directory or PYTHONPATH

    class SentimentInference:
        def __init__(self, config_path):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            self.config_path = config_path
            self.config_data = config_data
            # Adjust to access the correct key from the nested config structure
            self.model_hf_repo_id = config_data['model']['name_or_path'] 
            self.tokenizer_name_or_path = config_data['model'].get('tokenizer_name_or_path', self.model_hf_repo_id)
            self.local_model_weights_path = config_data['model'].get('local_model_weights_path', None) # Assuming it might be under 'model'
            self.load_from_local_pt = config_data['model'].get('load_from_local_pt', False)
            self.trust_remote_code_for_config = config_data['model'].get('trust_remote_code_for_config', True) # Default to True for custom code
            self.max_length = config_data['model']['max_length']
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

            try:
                if self.load_from_local_pt and self.local_model_weights_path:
                    print(f"Loading model from local path: {self.local_model_weights_path}")
                    # When loading local, config might also be local or from base model if not saved with custom checkpoint
                    # For simplicity, assume config is part of the saved pretrained local model or not strictly needed if all architecture is in code
                    self.config = AutoConfig.from_pretrained(self.local_model_weights_path, trust_remote_code=self.trust_remote_code_for_config)
                    self.model = ModernBertForSentiment.from_pretrained(self.local_model_weights_path, config=self.config, trust_remote_code=True)
                else:
                    print(f"Loading base ModernBertConfig from: {self.model_hf_repo_id}")
                    self.config = AutoConfig.from_pretrained(self.model_hf_repo_id, trust_remote_code=self.trust_remote_code_for_config)
                    print(f"Instantiating and loading model weights for {self.model_hf_repo_id} using ModernBertForSentiment...")
                    self.model = ModernBertForSentiment.from_pretrained(self.model_hf_repo_id, config=self.config, trust_remote_code=True)
                    print(f"Model {self.model_hf_repo_id} loaded successfully from Hugging Face Hub using ModernBertForSentiment.")
                self.model.to(self.device)
            except Exception as e:
                print(f"Failed to load model: {e}")
                # Optionally print more detailed traceback
                import traceback
                traceback.print_exc()
                exit()

            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, trust_remote_code=self.trust_remote_code_for_config)

        def print_debug_info(self):
            print(f"Model HF Repo ID: {self.model_hf_repo_id}")
            print(f"Tokenizer Name or Path: {self.tokenizer_name_or_path}")
            print(f"Local Model Weights Path: {self.local_model_weights_path}")
            print(f"Load from Local PT: {self.load_from_local_pt}")

    parser = argparse.ArgumentParser(description="Evaluate a sentiment analysis model on the IMDB test set.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="local_test_config.yaml",
        help="Path to the configuration file for SentimentInference (e.g., local_test_config.yaml or config.yaml)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation."
    )
    args = parser.parse_args()

    print(f"Using configuration: {args.config_path}")
    print("Loading sentiment model and tokenizer...")
    inferer = SentimentInference(config_path=args.config_path)
    model = inferer.model
    tokenizer = inferer.tokenizer
    max_length = inferer.max_length
    device = inferer.device

    print("Loading IMDB test dataset...")
    try:
        imdb_dataset_test = load_dataset("imdb", split="test")
    except Exception as e:
        print(f"Failed to load IMDB dataset: {e}")
        exit()

    print("Tokenizing dataset...")
    def tokenize_function(examples):
        tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
        tokenized_output["lengths"] = [sum(mask) for mask in tokenized_output["attention_mask"]]
        return tokenized_output
    
    tokenized_imdb_test = imdb_dataset_test.map(tokenize_function, batched=True)
    tokenized_imdb_test = tokenized_imdb_test.remove_columns(["text"])
    tokenized_imdb_test = tokenized_imdb_test.rename_column("label", "labels")
    tokenized_imdb_test.set_format("torch", columns=["input_ids", "attention_mask", "labels", "lengths"])

    test_dataloader = DataLoader(tokenized_imdb_test, batch_size=args.batch_size)
    
    print("Starting evaluation...")
    progress_bar = tqdm(evaluate(model, test_dataloader, device), desc="Evaluating")
    
    for update in progress_bar:
        if isinstance(update, dict):
            results = update
            break
        else:
            progress_bar.set_postfix_str(update)

    print("\n--- Evaluation Results ---")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key.capitalize()}: {value:.4f}")
        else:
            print(f"{key.capitalize()}: {value}")