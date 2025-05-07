from __future__ import annotations
import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, List
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from models import ModernBertForSentiment
from transformers import (
    AutoTokenizer,
    ModernBertConfig,
    ModernBertModel
)
from sklearn.metrics import accuracy_score, f1_score
from data_processing import download_and_prepare_datasets, create_dataloaders
from evaluation import evaluate
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR


def load_config(config_path="src/config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config_param): 
    config = config_param 
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    model_config = config['model']
    data_config = config['data'] 
    training_config = config['training']

    tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
    dsets = download_and_prepare_datasets(tokenizer, max_length=model_config['max_length'])
    train_dl, val_dl = create_dataloaders(dsets, tokenizer, training_config['batch_size'])

    bert_config = ModernBertConfig.from_pretrained(model_config['name'])
    bert_config.classifier_dropout = model_config['dropout']
    bert_config.num_labels = 1  # Ensure config has num_labels

    # Add pooling strategy and weighted layer config to bert_config
    bert_config.pooling_strategy = model_config.get('pooling_strategy', 'cls') # Default to 'cls' if not specified
    bert_config.num_weighted_layers = model_config.get('num_weighted_layers', 4) # Default if not specified

    # Add loss function configuration to bert_config
    # The model's __init__ expects a dict with 'name' and 'params'
    bert_config.loss_function = model_config.get('loss_function', {'name': 'SentimentWeightedLoss', 'params': {}})

    # Ensure output_hidden_states is True if using weighted layer pooling
    if bert_config.pooling_strategy in ['weighted_layer', 'cls_weighted_concat']:
        print(f"INFO: Setting output_hidden_states=True for {bert_config.pooling_strategy} pooling.")
        bert_config.output_hidden_states = True
    else:
        # Explicitly set to False if not needed, though default might be False
        bert_config.output_hidden_states = False 

    # 1. Load the pre-trained base BERT model
    print("Loading pre-trained base ModernBertModel...")
    base_bert_model = ModernBertModel.from_pretrained(
        model_config['name'],
        config=bert_config # Pass config if needed for architecture consistency
    )

    # 2. Instantiate the custom model wrapper from config ONLY
    # This initializes the structure including the classifier head, but doesn't load bert weights
    print("Instantiating custom model structure...")
    model = ModernBertForSentiment(config=bert_config)

    # 3. Manually assign the loaded pre-trained bert model to the custom model's bert attribute
    print("Assigning pre-trained base model to custom model...")
    model.bert = base_bert_model

    # Now, model.bert has pre-trained weights, and model.classifier is randomly initialized.
    model.to(device)

    optimizer = AdamW(
        model.parameters(), 
        lr=float(training_config['lr']), 
        weight_decay=float(training_config['weight_decay_rate'])
    )
    lr_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=training_config['epochs'] * len(train_dl),
    )

    best_f1 = 0.0
    start_epoch = 1

    resume_checkpoint_path = training_config.get('resume_from_checkpoint')
    optimizer_state_to_load = None
    scheduler_state_to_load = None

    # Preserve the current session's output_dir in case config is overridden by checkpoint's config
    # This allows current CLI/YAML to specify a *new* output location for checkpoints from this resumed run.
    current_session_output_dir = config['model'].get('output_dir')

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        try:
            checkpoint_content = torch.load(resume_checkpoint_path, map_location=device)
            
            parsed_epoch_from_filename = None
            filename_for_parse = os.path.basename(resume_checkpoint_path)
            match = re.search(r'_epoch(\d+)|[._-]e(\d+)', filename_for_parse, re.IGNORECASE)
            if match:
                epoch_str = match.group(1) or match.group(2)
                if epoch_str: parsed_epoch_from_filename = int(epoch_str)

            if isinstance(checkpoint_content, dict) and 'model_state_dict' in checkpoint_content:
                print("Checkpoint is new format (dictionary).")
                
                # Check if checkpoint contains its own config
                if 'config' in checkpoint_content:
                    print("INFO: Found 'config' in checkpoint. This will be used for the resumed session.")
                    loaded_config_from_checkpoint = checkpoint_content['config']
                    config = loaded_config_from_checkpoint # Override current config with checkpoint's config
                    
                    # Restore the current session's output_dir if it was set, allowing override
                    if current_session_output_dir:
                        config['model']['output_dir'] = current_session_output_dir
                    
                    # Clear the resume_from_checkpoint path from the now-active config
                    # This resume operation has been 'consumed'. Future saves of this session should
                    # not try to re-resume from this same old checkpoint path
                    # unless specifically entered in config.yaml
                    if 'training' in config and 'resume_from_checkpoint' in config['training']:
                        config['training']['resume_from_checkpoint'] = ""
                    print(f"Resumed session will use config from checkpoint. Effective LR: {config.get('training', {}).get('lr', 'N/A')}, Total Target Epochs: {config.get('training', {}).get('epochs', 'N/A')}.")
                else:
                    print("INFO: Checkpoint is new format but does not contain a 'config' entry. Using current session's config.")

                model.load_state_dict(checkpoint_content['model_state_dict'])
                optimizer_state_to_load = checkpoint_content.get('optimizer_state_dict')
                scheduler_state_to_load = checkpoint_content.get('scheduler_state_dict')
                best_f1 = checkpoint_content.get('best_f1', best_f1) # Update best_f1 if present
                
                epoch_from_data = checkpoint_content.get('epoch')
                if epoch_from_data is not None:
                    start_epoch = epoch_from_data + 1
                    print(f"Epoch from data: {epoch_from_data}. Resuming: epoch {start_epoch}.")
                elif parsed_epoch_from_filename is not None:
                    start_epoch = parsed_epoch_from_filename + 1
                    print(f"Epoch from data missing. Epoch from filename: {parsed_epoch_from_filename}. Resuming: epoch {start_epoch}.")
                else:
                    print(f"WARNING: Epoch not in data or filename. Defaulting start_epoch: {start_epoch}.")
            else: # Old format (checkpoint_content is the state_dict)
                print("Checkpoint is old format (direct state_dict). Config from checkpoint not available. Using current session's config.")
                model.load_state_dict(checkpoint_content)
                print("INFO: Old format. Optimizer, scheduler, best_f1 not loaded from file.")
                if parsed_epoch_from_filename is not None:
                    start_epoch = parsed_epoch_from_filename + 1
                    print(f"Epoch from filename: {parsed_epoch_from_filename}. Resuming: epoch {start_epoch}.")
                else:
                    print(f"WARNING: Old format & epoch not in filename. Defaulting start_epoch: {start_epoch}.")

            print(f"Checkpoint processing complete. Effective start epoch: {start_epoch}. Current best F1: {best_f1:.4f}")

            if optimizer_state_to_load:
                optimizer.load_state_dict(optimizer_state_to_load)
                print("Optimizer state loaded.")
            if scheduler_state_to_load:
                lr_scheduler.load_state_dict(scheduler_state_to_load)
                print("Scheduler state loaded.")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Proceeding with training from scratch (epoch 1, best_f1 0.0).")
            start_epoch = 1 # Ensure defaults on error
            best_f1 = 0.0   # Ensure defaults on error
            optimizer_state_to_load = None # Ensure no partial load attempts
            scheduler_state_to_load = None
    elif resume_checkpoint_path:
        print(f"WARNING: Checkpoint path '{resume_checkpoint_path}' provided but not found. Starting training from scratch.")

    # Re-derive config sections from the potentially updated 'config' object
    model_config = config['model']
    data_config = config['data'] 
    training_config = config['training']
    
    # From here, use model_config, data_config, training_config

    optimizer = AdamW(
        model.parameters(), 
        lr=float(training_config['lr']), 
        weight_decay=float(training_config['weight_decay_rate'])
    )
    lr_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=training_config['epochs'] * len(train_dl), # Uses epochs from current/checkpoint config
    )

    # Load optimizer and scheduler states if they were retrieved from checkpoint
    if optimizer_state_to_load:
        try:
            optimizer.load_state_dict(optimizer_state_to_load)
            print("Optimizer state successfully loaded.")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}. Optimizer reinitialized.")
    if scheduler_state_to_load:
        try:
            lr_scheduler.load_state_dict(scheduler_state_to_load)
            print("Scheduler state successfully loaded.")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}. Scheduler reinitialized.")

    # The loop runs from determined start_epoch up to the total_epochs from the active config
    for epoch in range(start_epoch, training_config['epochs'] + 1):
        model.train()
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss # Get loss directly from the output object

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print(f"Epoch {epoch} | Step {step}/{len(train_dl)} | Training Loss {loss.item():.4f}")

        metrics = evaluate(model, val_dl, device)
        print(f"Epoch {epoch} validation – Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
              f"AUC: {metrics['roc_auc']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, MCC: {metrics['mcc']:.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            # Construct filename with pooling strategy, epoch, accuracy, and f1
            pooling_str = model_config.get('pooling_strategy', 'cls')
            ckpt_filename = f"{pooling_str}_epoch{epoch}_{metrics['accuracy']:.4f}acc_{metrics['f1']:.4f}f1.pt"
            output_dir = Path(model_config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = output_dir / ckpt_filename
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_f1': best_f1,
                'config': config # Save the active config (could be from checkpoint or current run)
            }
            torch.save(checkpoint_data, ckpt_path)
            print(f"✨ Saved new best model to {ckpt_path}")


if __name__ == "__main__":
    
    # Load config
    config = load_config()
    
    # Passed args override config.yaml
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=None, type=str, help="HF ModernBERT checkpoint")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay_rate", type=float, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--output_dir", default=None, type=str)
    p.add_argument("--resume_from_checkpoint", default=None, type=str, help="Path to checkpoint file to resume training from.")
    args = p.parse_args()

    # Update config with passed args
    config['model']['name'] = args.model_name or config['model']['name']
    config['training']['epochs'] = args.epochs or config['training']['epochs']
    config['training']['batch_size'] = args.batch_size or config['training']['batch_size']
    config['training']['lr'] = args.lr or config['training']['lr']
    config['training']['weight_decay_rate'] = args.weight_decay_rate or config['training']['weight_decay_rate']
    config['model']['max_length'] = args.max_length or config['model']['max_length']
    config['model']['output_dir'] = args.output_dir or config['model']['output_dir']
    config['training']['resume_from_checkpoint'] = args.resume_from_checkpoint or config['training'].get('resume_from_checkpoint')
    
    train(config)
