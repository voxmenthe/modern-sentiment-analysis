import json
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, ModernBertConfig, DebertaV2Config
from src.data import download_and_prepare_datasets, create_dataloaders
from src.modeling import ModernBertForSentiment, DebertaForSentiment
from src.utils import generate_artifact_name, parse_artifact_filename
import datetime
import re

def load_config(config_path: str = "src/config.yaml") -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_metrics(config: dict, metrics_path: str, output_dir: Path = Path("plots")):
    """Plot training and validation metrics from JSON history."""
    metrics_file_path = Path(metrics_path)
    with open(metrics_file_path, 'r') as f:
        history = json.load(f)
    epochs_data = history.get("epoch", [])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse the metrics filename to get parts for new plot names
    # We need model_name, loss_function_name from config, and timestamp, epoch from metrics_filename
    parsed_metrics_name = parse_artifact_filename(metrics_file_path.name)
    if not parsed_metrics_name:
        print(f"WARNING: Could not parse metrics filename '{metrics_file_path.name}' for plot naming. Using generic plot names.")
        # Fallback to old behavior or simplified naming if parsing fails
        # For now, we'll let it try to save with potentially missing parts or use defaults in generate_artifact_name
        # but ideally, this should be robust.
        # Fallback values (less ideal as they might not match other artifacts)
        timestamp_for_plot = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # Use last epoch from history if parsing fails, or a default if history is also empty
        epoch_for_plot = epochs_data[-1] if epochs_data else config.get('training', {}).get('epochs', 0) 
    else:
        timestamp_for_plot = parsed_metrics_name['timestamp']
        epoch_for_plot = parsed_metrics_name['epoch'] # Epoch from filename is likely last training epoch

    model_config_name = config['model']['name']
    loss_function_name_from_config = config['model']['loss_function']['name']

    plot_types = {
        "loss": {"train_key": "train_loss", "val_key": "val_loss", "label": "Loss", "title_suffix": "over Epochs"},
        "accuracy": {"train_key": "train_accuracy", "val_key": "val_accuracy", "label": "Accuracy", "title_suffix": "over Epochs"},
        "f1_score": {"train_key": "train_f1", "val_key": "val_f1", "label": "F1 Score", "title_suffix": "over Epochs"}
    }

    for plot_key, plot_config in plot_types.items():
        plt.figure(figsize=(10, 6))
        if plot_config["train_key"] in history:
            plt.plot(epochs_data, history.get(plot_config["train_key"]), label=f'Train {plot_config["label"]}')
        if plot_config["val_key"] in history:
            plt.plot(epochs_data, history.get(plot_config["val_key"]), label=f'Val {plot_config["label"]}')
        plt.xlabel('Epoch')
        plt.ylabel(plot_config["label"])
        plt.title(f'{plot_config["label"]} {plot_config["title_suffix"]}')
        plt.legend()
        
        plot_filename_path = generate_artifact_name(
            base_output_dir=output_dir,
            model_config_name=model_config_name, # generate_artifact_name will take care of .split('/')[-1]
            loss_function_name=loss_function_name_from_config,
            epoch=epoch_for_plot, # Use epoch from parsed metrics filename
            artifact_type=f"plot_{plot_key}_curve", # e.g., plot_loss_curve
            timestamp_str=timestamp_for_plot, # Use timestamp from parsed metrics filename
            extension="png"
        )
        plt.savefig(plot_filename_path)
        print(f"Saved plot to {plot_filename_path}")
        plt.close()

def plot_confusion_matrix(config_path: str, checkpoint_path: str, output_dir: Path = Path("plots")):
    """Compute and plot confusion matrix using the model checkpoint and validation dataset."""
    print(f"DEBUG: plot_confusion_matrix called with config_path: {config_path}") # Diagnostic print
    config = load_config(config_path)
    model_cfg = config['model']
    print(f"DEBUG: Loaded model_cfg: {model_cfg}") # Diagnostic print
    training_cfg = config['training']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = model_cfg.get('model_type', 'modernbert').lower()
    print(f"DEBUG: Determined model_type: '{model_type}'") # Diagnostic print

    # Tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['name'])
    datasets = download_and_prepare_datasets(tokenizer, max_length=model_cfg.get('max_length', 256))
    _, val_dl = create_dataloaders(datasets, tokenizer, training_cfg.get('batch_size', 16))

    # Model setup
    if model_type == 'modernbert':
        m_config = ModernBertConfig.from_pretrained(model_cfg['name'])
        m_config.classifier_dropout = model_cfg.get('dropout', 0.1)
        m_config.num_labels = 1
        m_config.pooling_strategy = model_cfg.get('pooling_strategy', 'cls')
        m_config.num_weighted_layers = model_cfg.get('num_weighted_layers', 4)
        m_config.loss_function = model_cfg.get('loss_function', {})
        if m_config.pooling_strategy in ['weighted_layer', 'cls_weighted_concat']:
            m_config.output_hidden_states = True
        else:
            m_config.output_hidden_states = False
        
        print("DEBUG: Attempting to initialize ModernBertForSentiment...") # Diagnostic print
        model = ModernBertForSentiment(config=m_config)

    elif model_type == 'deberta':
        d_config = DebertaV2Config.from_pretrained(model_cfg['name'])
        d_config.classifier_dropout = model_cfg.get('dropout', 0.1)
        d_config.num_labels = 1
        d_config.pooling_strategy = model_cfg.get('pooling_strategy', 'cls')
        d_config.num_weighted_layers = model_cfg.get('num_weighted_layers', 4)
        d_config.loss_function = model_cfg.get('loss_function', {})
        if d_config.pooling_strategy in ['weighted_layer', 'cls_weighted_concat']:
            d_config.output_hidden_states = True
        else:
            d_config.output_hidden_states = False

        print("DEBUG: Attempting to initialize DebertaForSentiment...") # Diagnostic print
        model = DebertaForSentiment(config=d_config)

    else:
        print(f"ERROR: Unsupported model_type '{model_type}' in config '{config_path}'. Supported: 'modernbert', 'deberta'.")
        return

    # Load checkpoint file and apply state_dict directly
    cp_path = Path(checkpoint_path)
    if not cp_path.is_file():
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        return
    checkpoint = torch.load(cp_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    print(f"DEBUG: Model object type before loading weights: {type(model)}")
    model.to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"WARNING: Missing keys when loading state_dict: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"WARNING: Unexpected keys in state_dict: {incompatible.unexpected_keys}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            if logits.shape[1] > 1:
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().squeeze()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch['labels'].cpu().numpy().tolist())

    # For naming the plot:
    # Parse checkpoint filename for its components
    parsed_checkpoint_name = parse_artifact_filename(Path(checkpoint_path).name)
    
    current_config = load_config(config_path) # Load config for model and loss name
    model_cfg_name_for_plot = current_config['model']['name']
    loss_func_name_for_plot = current_config['model']['loss_function']['name']
    
    timestamp_for_cm = datetime.datetime.now().strftime("%Y%m%d%H%M%S") # Default timestamp
    epoch_for_cm = 0 # Default epoch
    f1_for_cm = None # Default F1

    if parsed_checkpoint_name:
        timestamp_for_cm = parsed_checkpoint_name['timestamp']
        epoch_for_cm = parsed_checkpoint_name['epoch']
        f1_for_cm = parsed_checkpoint_name['f1_score'] # This is the F1 from training
        # Note: model_name from parsed_checkpoint_name can also be used/verified against config
    else:
        print(f"WARNING: Could not parse checkpoint filename '{Path(checkpoint_path).name}' for confusion matrix plot naming. Using defaults/fallbacks.")
        # Attempt to get epoch from checkpoint data if parsing fails (might be in old format checkpoint)
        # This part depends on the structure of 'checkpoint' loaded earlier
        # For now, we rely on parsing or defaults.
        # Checkpoint data might contain 'epoch' and 'best_f1'
        try:
            # Checkpoint is loaded around line 100-102 in the original file
            # If it's a dictionary and has the keys:
            # Ensure 'checkpoint' variable is accessible here or re-load if necessary.
            # This function re-loads the checkpoint anyway, so we can access it.
            # The reloaded checkpoint for CM is: `checkpoint = torch.load(cp_path, map_location=device)`
            # Let's assume `checkpoint` variable is the loaded content.
            # Need to ensure `checkpoint` is defined from the loaded data.
            # The code snippet shows `checkpoint = torch.load(cp_path, map_location=device)`
            # So, checkpoint is the loaded data.
            loaded_checkpoint_content = torch.load(Path(checkpoint_path), map_location=torch.device("cpu")) # Load fresh to inspect
            if isinstance(loaded_checkpoint_content, dict):
                epoch_for_cm = loaded_checkpoint_content.get('epoch', epoch_for_cm)
                f1_for_cm = loaded_checkpoint_content.get('best_f1', f1_for_cm)
        except Exception as e:
            print(f"INFO: Could not inspect checkpoint content for fallback epoch/f1 for plot naming: {e}")

    cm = confusion_matrix(all_labels, all_preds)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    cm_filename_path = generate_artifact_name(
        base_output_dir=output_dir,
        model_config_name=model_cfg_name_for_plot, # from config
        loss_function_name=loss_func_name_for_plot, # from config
        epoch=epoch_for_cm, # from parsed ckpt name or fallback
        artifact_type="plot_confusion_matrix",
        f1_score=f1_for_cm, # from parsed ckpt name or fallback
        timestamp_str=timestamp_for_cm, # from parsed ckpt name or fallback
        extension="png"
    )
    plt.savefig(cm_filename_path)
    print(f"Saved confusion matrix to {cm_filename_path}")
    plt.close()

def main():
    """CLI entrypoint to plot metrics and confusion matrix."""
    import argparse
    parser = argparse.ArgumentParser(description='Visualize training metrics and confusion matrix')
    parser.add_argument('--metrics-file', type=str, default='checkpoints/metrics.json', help='Path to metrics JSON file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='src/config.yaml', help='Path to config YAML')
    parser.add_argument('--output-dir', type=str, default='plots', help='Directory to save plots')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    config_main = load_config(args.config) # Load config once for main

    # Pass the loaded config to plot_metrics
    plot_metrics(config_main, args.metrics_file, out_dir)
    
    if args.checkpoint:
        # plot_confusion_matrix loads its own config, but args.config is the same path
        plot_confusion_matrix(args.config, args.checkpoint, out_dir)

if __name__ == '__main__':
    main()
