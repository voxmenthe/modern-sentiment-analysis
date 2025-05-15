from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader


def download_and_prepare_datasets(tokenizer: AutoTokenizer, max_length: int = 256) -> DatasetDict:
    """Download IMDB via HF Datasets and tokenize"""
    raw_dset = load_dataset("imdb")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = raw_dset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")
    return tokenized


def create_dataloaders(dset: DatasetDict, tokenizer: AutoTokenizer, batch_size: int = 16):
    def add_len(example):
        example["lengths"] = (example["attention_mask"].sum()).long()
        return example

    dset = dset.map(add_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_dl = DataLoader(
        dset["train"], 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collator,
        num_workers=58,
        pin_memory=True,
        persistent_workers=True
        )
    val_dl = DataLoader(
        dset["test"], 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collator,
        num_workers=58,
        pin_memory=True,
        persistent_workers=True
        )
    return train_dl, val_dl
