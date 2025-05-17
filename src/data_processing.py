from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import os
import torch


def download_and_prepare_datasets(tokenizer: AutoTokenizer, max_length: int = 256) -> DatasetDict:
    """Download IMDB via HF Datasets and tokenize"""
    raw_dset = load_dataset("imdb")

    def tokenize_fn(batch):
        # Use padding="max_length" for more efficient processing
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"  # Pre-pad during tokenization
        )

    tokenized = raw_dset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=max(1, os.cpu_count() // 2)  # Parallelize tokenization
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")

    # Pre-compute lengths during dataset preparation
    tokenized = tokenized.map(
        lambda example: {"lengths": example["attention_mask"].sum(dim=1).long()},
        batched=True,
        batch_size=1000,
        num_proc=max(1, os.cpu_count() // 2)
    )

    return tokenized


def create_dataloaders(dset: DatasetDict, tokenizer: AutoTokenizer, batch_size: int = 16):

    # Optimize for GPU efficiency by padding to multiples of 8
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
        pad_to_multiple_of=8
    )

    # Determine optimal number of workers based on system
    num_workers = min(os.cpu_count() - 1, 8) if os.cpu_count() > 1 else 0

    train_dl = DataLoader(
        dset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
        )
    val_dl = DataLoader(
        dset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
        )
    return train_dl, val_dl
