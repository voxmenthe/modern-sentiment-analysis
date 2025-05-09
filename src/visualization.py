import json
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel
from src.data_processing import download_and_prepare_datasets, create_dataloaders
from src.models import ModernBertForSentiment

def load_config(config_path: str = "src/config.yaml") -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_metrics(metrics_path: str, output_dir: Path = Path("plots")):
    """Plot training and validation metrics from JSON history."""
    metrics_path = Path(metrics_path)
    with open(metrics_path, 'r') as f:
        history = json.load(f)
    epochs = history.get("epoch", [])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.get("train_loss", []), label='Train Loss')
    plt.plot(epochs, history.get("val_loss", []), label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig(output_dir / 'loss.png')
    plt.close()

    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.get("train_accuracy", []), label='Train Acc')
    plt.plot(epochs, history.get("val_accuracy", []), label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig(output_dir / 'accuracy.png')
    plt.close()

    # F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.get("train_f1", []), label='Train F1')
    plt.plot(epochs, history.get("val_f1", []), label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over Epochs')
    plt.legend()
    plt.savefig(output_dir / 'f1_score.png')
    plt.close()

def plot_confusion_matrix(config_path: str, checkpoint_path: str, output_dir: Path = Path("plots")):
    """Compute and plot confusion matrix using the model checkpoint and validation dataset."""
    config = load_config(config_path)
    model_cfg = config['model']
    training_cfg = config['training']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['name'])
    datasets = download_and_prepare_datasets(tokenizer, max_length=model_cfg.get('max_length', 256))
    _, val_dl = create_dataloaders(datasets, tokenizer, training_cfg.get('batch_size', 16))

    # Model setup
    bert_cfg = ModernBertConfig.from_pretrained(model_cfg['name'])
    bert_cfg.classifier_dropout = model_cfg.get('dropout', 0.1)
    bert_cfg.num_labels = 1
    bert_cfg.pooling_strategy = model_cfg.get('pooling_strategy', 'cls')
    bert_cfg.num_weighted_layers = model_cfg.get('num_weighted_layers', 4)
    bert_cfg.loss_function = model_cfg.get('loss_function', {})
    if bert_cfg.pooling_strategy in ['weighted_layer', 'cls_weighted_concat']:
        bert_cfg.output_hidden_states = True
    else:
        bert_cfg.output_hidden_states = False

    base_model = ModernBertModel.from_pretrained(model_cfg['name'], config=bert_cfg)
    model = ModernBertForSentiment(config=bert_cfg)
    model.bert = base_model

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Collect predictions
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

    cm = confusion_matrix(all_labels, all_preds)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix.png')
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
    plot_metrics(args.metrics_file, out_dir)
    if args.checkpoint:
        plot_confusion_matrix(args.config, args.checkpoint, out_dir)

if __name__ == '__main__':
    main()
