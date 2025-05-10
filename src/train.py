from __future__ import annotations
import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, List
import yaml
import json
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from src.models import ModernBertForSentiment, DebertaForSentiment
from transformers import (
    AutoTokenizer,
    ModernBertConfig,
    ModernBertModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    DebertaV2Model,
    DebertaV2Config
)
from sklearn.metrics import accuracy_score, f1_score
from src.data_processing import download_and_prepare_datasets, create_dataloaders
from src.evaluation import evaluate
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from src.utils import generate_artifact_name
try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None # type: ignore


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

    model_type = model_config.get('model_type', 'modernbert') # Get model_type, default to modernbert

    if model_type == 'deberta':
        print(f"INFO: Explicitly loading SLOW DebertaV2Tokenizer for {model_config['name']}.")
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_config['name'])
    elif model_type == 'modernbert':
        print(f"INFO: Loading tokenizer for ModernBERT model: {model_config['name']} using AutoTokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
    else:
        print(f"WARNING: Unknown model_type '{model_type}'. Falling back to AutoTokenizer with use_fast=False.")
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'], use_fast=False)

    dsets = download_and_prepare_datasets(tokenizer, max_length=model_config['max_length'])
    train_dl, val_dl = create_dataloaders(dsets, tokenizer, training_config['batch_size'])

    if model_type == 'deberta':
        print(f"Loading custom DebertaForSentiment model: {model_config['name']}")
        # 1. Load DeBERTa config and customize it
        deberta_base_config = DebertaV2Config.from_pretrained(model_config['name'])
        deberta_base_config.num_labels = 1 # For sentiment regression-style output
        deberta_base_config.classifier_dropout = model_config.get('dropout', 0.1) # Use model_config dropout
        
        # Add pooling strategy and weighted layer config
        deberta_base_config.pooling_strategy = model_config.get('pooling_strategy', 'cls')
        deberta_base_config.num_weighted_layers = model_config.get('num_weighted_layers', 4)
        deberta_base_config.loss_function = model_config.get('loss_function', {'name': 'SentimentWeightedLoss', 'params': {}})

        if deberta_base_config.pooling_strategy in ['weighted_layer', 'cls_weighted_concat']:
            print(f"INFO: Setting output_hidden_states=True for {deberta_base_config.pooling_strategy} pooling (DeBERTa).")
            deberta_base_config.output_hidden_states = True
        else:
            deberta_base_config.output_hidden_states = False

        # 2. Load the pre-trained base DeBERTa model with potentially modified config (for output_hidden_states)
        print("Loading pre-trained base DebertaV2Model...")
        base_deberta_model = DebertaV2Model.from_pretrained(
            model_config['name'],
            config=deberta_base_config 
        )

        # 3. Instantiate the custom DebertaForSentiment model wrapper using the full config
        print("Instantiating custom DebertaForSentiment model structure...")
        model = DebertaForSentiment(config=deberta_base_config) # Pass the full config here

        # 4. Manually assign the loaded pre-trained deberta model to the custom model's deberta attribute
        print("Assigning pre-trained base model to custom DeBERTa model...")
        model.deberta = base_deberta_model
        print("Custom DebertaForSentiment model loaded and configured.")

    elif model_type == 'modernbert':
        bert_config = ModernBertConfig.from_pretrained(model_config['name'])
        bert_config.classifier_dropout = model_config['dropout']
        bert_config.num_labels = 1  # Ensure config has num_labels

        bert_config.pooling_strategy = model_config.get('pooling_strategy', 'cls')
        bert_config.num_weighted_layers = model_config.get('num_weighted_layers', 4)
        bert_config.loss_function = model_config.get('loss_function', {'name': 'SentimentWeightedLoss', 'params': {}})

        if bert_config.pooling_strategy in ['weighted_layer', 'cls_weighted_concat']:
            print(f"INFO: Setting output_hidden_states=True for {bert_config.pooling_strategy} pooling (ModernBERT).")
            bert_config.output_hidden_states = True
        else:
            bert_config.output_hidden_states = False

        print(f"Loading ModernBERT model: {model_config['name']}")
        # 1. Load the pre-trained base BERT model
        print("Loading pre-trained base ModernBertModel...")
        base_bert_model = ModernBertModel.from_pretrained(
            model_config['name'],
            config=bert_config # Pass config if needed for architecture consistency
        )

        # 2. Instantiate the custom model wrapper from config ONLY
        print("Instantiating custom ModernBertForSentiment model structure...")
        model = ModernBertForSentiment(config=bert_config)

        # 3. Manually assign the loaded pre-trained bert model to the custom model's bert attribute
        print("Assigning pre-trained base model to custom model...")
        model.bert = base_bert_model
        print("ModernBERT model loaded and configured.")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'modernbert' or 'deberta'.")

    model.to(device)

    # Apply torch.compile()
    if torch.backends.mps.is_available() and hasattr(torch, 'compile'):
        print("Attempting to compile the model with torch.compile() for MPS (backend: aot_eager)...")
        try:
            model = torch.compile(model, backend="aot_eager")
            print("Model compiled successfully for MPS with 'aot_eager' backend.")
        except Exception as e_mps_aot:
            print(f"MPS model compilation failed with 'aot_eager' backend: {e_mps_aot}. Proceeding without compilation on MPS.")
    elif torch.cuda.is_available() and hasattr(torch, 'compile'):
        print("Attempting to compile the model with torch.compile() for CUDA (mode: default)...")
        try:
            model = torch.compile(model, mode="default") 
            print("Model compiled successfully for CUDA.")
        except Exception as e_cuda:
            print(f"CUDA model compilation failed: {e_cuda}. Proceeding without compilation.")
    elif not hasattr(torch, 'compile'):
        print("torch.compile not available in this PyTorch version.")

    # Gradient Checkpointing (optional, from config)
    if training_config.get("gradient_checkpointing", False): # Default to False if not specified
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled.")
        else:
            print("Warning: Model does not support gradient_checkpointing_enable().")

    # Optimizer and LR Scheduler will be initialized *after* potential checkpoint loading
    # to use the most up-to-date config (e.g., LR from checkpoint).
    optimizer = None
    lr_scheduler = None
    # scaler for AMP will also be initialized after device selection and config finalization
    scaler = None
    if device.type == 'cuda' and GradScaler is not None:
        scaler = GradScaler()
        print("CUDA GradScaler initialized.")
    elif device.type == 'cuda' and GradScaler is None:
        print("Warning: torch.cuda.amp.GradScaler not found, CUDA AMP will not use GradScaler.")

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
    
    # Initialize optimizer and scheduler here, using the final configuration
    # (potentially updated from checkpoint)
    print(f"Initializing optimizer with LR: {float(training_config['lr'])} and Weight Decay: {float(training_config['weight_decay_rate'])}")
    optimizer = AdamW(
        model.parameters(), 
        lr=float(training_config['lr']), 
        weight_decay=float(training_config['weight_decay_rate']),
        fused=True if device.type == 'cuda' else False # Use fused AdamW on CUDA if available
    )
    print(f"Initializing LinearLR scheduler with total_iters: {training_config['epochs'] * len(train_dl)}")
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

    history = {
        "epoch": [],
        "train_loss": [], "train_accuracy": [], "train_f1": [], "train_roc_auc": [], "train_precision": [], "train_recall": [], "train_mcc": [],
        "val_loss": [], "val_accuracy": [], "val_f1": [], "val_roc_auc": [], "val_precision": [], "val_recall": [], "val_mcc": []
    }

    # The loop runs from determined start_epoch up to the total_epochs from the active config
    for epoch in range(start_epoch, training_config['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}

            current_batch_for_model = batch
            optimizer.zero_grad(set_to_none=True) # Use set_to_none=True


            if device.type == 'mps':
                outputs = model(**current_batch_for_model) 
                loss = outputs.loss                         
                loss.backward()
                optimizer.step()
            elif device.type == 'cuda' and scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16): 
                    outputs = model(**current_batch_for_model)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # CPU or CUDA without GradScaler
                outputs = model(**current_batch_for_model)
                loss = outputs.loss
                loss.backward()
                optimizer.step()


            lr_scheduler.step()
            # optimizer.zero_grad() # Moved and changed to set_to_none=True above
            if loss is not None: # Add to total_loss only if loss is not None
                 total_loss += loss.item()
            if step % 100 == 0:
                # ---- START DETAILED DEBUG PRINT ----
                print(f"DEBUG PRINT BEFORE AVG LOSS: Epoch {epoch}, Step {step}, total_loss: {total_loss:.8f}, step: {step}")
                # ----  END DETAILED DEBUG PRINT  ----
                print(f"Epoch {epoch} | Step {step}/{len(train_dl)} | Training Loss {total_loss/step:.4f}")

        # Compute and log training metrics
        train_metrics = evaluate(model, train_dl, device)
        print(f"Epoch {epoch} train – Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, "
              f"AUC: {train_metrics['roc_auc']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, MCC: {train_metrics['mcc']:.4f}")
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["train_roc_auc"].append(train_metrics["roc_auc"])
        history["train_precision"].append(train_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["train_mcc"].append(train_metrics["mcc"])

        metrics = evaluate(model, val_dl, device)
        print(f"Epoch {epoch} validation – Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
              f"AUC: {metrics['roc_auc']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, MCC: {metrics['mcc']:.4f}")
        # Record validation metrics
        history["val_loss"].append(metrics["loss"])
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_f1"].append(metrics["f1"])
        history["val_roc_auc"].append(metrics["roc_auc"])
        history["val_precision"].append(metrics["precision"])
        history["val_recall"].append(metrics["recall"])
        history["val_mcc"].append(metrics["mcc"])
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            
            # Generate checkpoint path using the new utility function
            # A timestamp for this specific save operation
            current_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            ckpt_path = generate_artifact_name(
                base_output_dir=model_config['output_dir'],
                model_config_name=model_config['name'],
                loss_function_name=config['model']['loss_function']['name'],
                epoch=epoch,
                artifact_type="checkpoint",
                f1_score=best_f1,
                timestamp_str=current_timestamp,
                extension="pt"
            )
            
            output_dir = Path(model_config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_f1': best_f1,
                'config': config
            }
            torch.save(checkpoint_data, ckpt_path)
            print(f"✨ Saved new best model to {ckpt_path}")

    # After training, save metrics history
    # Generate metrics filename using the new utility function
    # Use the final epoch from training_config for the metrics file, and a new timestamp
    final_epoch_for_metrics = training_config['epochs']
    metrics_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    metrics_file_path = generate_artifact_name(
        base_output_dir=model_config['output_dir'],
        model_config_name=model_config['name'],
        loss_function_name=config['model']['loss_function']['name'],
        epoch=final_epoch_for_metrics,
        artifact_type="metrics",
        timestamp_str=metrics_timestamp,
        extension="json"
    )
    
    # Ensure output directory for metrics exists (though generate_artifact_name places it in base_output_dir)
    Path(model_config['output_dir']).mkdir(parents=True, exist_ok=True)

    with open(metrics_file_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Metrics history saved to {metrics_file_path}")


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
    p.add_argument("--gradient_checkpointing", type=bool, default=None, help="Enable gradient checkpointing.")
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
    if args.gradient_checkpointing is not None:
        config['training']['gradient_checkpointing'] = args.gradient_checkpointing
    
    train(config)
