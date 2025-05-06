from __future__ import annotations
import argparse
import math
import os
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
    ModernBertConfig
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


def train(config):
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
    model = ModernBertForSentiment.from_pretrained(
        model_config['name'],
        config=bert_config
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=training_config['lr'], weight_decay=training_config['weight_decay_rate'])
    lr_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=training_config['epochs'] * len(train_dl),
    )

    best_f1 = 0.0
    for epoch in range(1, training_config['epochs'] + 1):
        model.train()
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print(f"Epoch {epoch} | Step {step}/{len(train_dl)} | Loss {loss.item():.4f}")

        metrics = evaluate(model, val_dl, device)
        print(f"Epoch {epoch} validation – acc: {metrics['accuracy']:.4f}  f1: {metrics['f1']:.4f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            ckpt_path = Path(model_config['output_dir']) / "best_model.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"✨ Saved new best model to {ckpt_path}")


# ---------------------------
# CLI Entrypoint
# ---------------------------

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
    args = p.parse_args()

    # Update config with passed args
    config['model']['name'] = args.model_name or config['model']['name']
    config['training']['epochs'] = args.epochs or config['training']['epochs']
    config['training']['batch_size'] = args.batch_size or config['training']['batch_size']
    config['training']['lr'] = args.lr or config['training']['lr']
    config['training']['weight_decay_rate'] = args.weight_decay_rate or config['training']['weight_decay_rate']
    config['model']['max_length'] = args.max_length or config['model']['max_length']
    config['model']['output_dir'] = args.output_dir or config['model']['output_dir']
    
    train(config)
