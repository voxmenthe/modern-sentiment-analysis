from datasets import load_dataset, DatasetDict, Dataset
import datasets # Add this for datasets.Value
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import os
import torch
import yaml # Added for loading config

# Import augmentation function
from .data_augmentation import apply_char_truncation_and_adjust_label


def download_and_prepare_datasets(
    tokenizer: AutoTokenizer, 
    tokenizer_max_length: int = 256, 
    config_path: str = "src/config.yaml",
    raw_dset_override: DatasetDict | None = None  # Added for testing
) -> DatasetDict:
    """Download IMDB via HF Datasets, optionally augment, and tokenize"""
    
    # Load augmentation configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_aug_config = config.get('data', {})
    enable_augmentation = data_aug_config.get('enable_augmentation', False)
    truncation_percentage = data_aug_config.get('truncation_percentage', 0.8)
    truncation_methods = data_aug_config.get('truncation_method', []) # Ensure it's a list
    apply_to_split = data_aug_config.get('apply_to_split', 'train')

    # Load the full raw dataset first
    if raw_dset_override is not None:
        full_raw_dset = raw_dset_override
    else:
        full_raw_dset = load_dataset("imdb")

    # Ensure all label columns are initially represented as lists of Python floats
    # This simplifies downstream processing and ensures consistency before tokenization and set_format.
    for split_name_to_prepare in full_raw_dset.keys():
        current_split_for_prep = full_raw_dset[split_name_to_prepare]
        if 'label' in current_split_for_prep.column_names:
            def cast_label_to_float(examples):
                return {'label': [float(l) for l in examples['label']]}
            full_raw_dset[split_name_to_prepare] = current_split_for_prep.map(
                cast_label_to_float, 
                batched=True
            )

    if enable_augmentation and truncation_methods and apply_to_split in full_raw_dset.keys():
        raw_dset_split_to_augment = full_raw_dset[apply_to_split]
        
        # Labels in raw_dset_split_to_augment are now already float due to the loop above
        all_texts = list(raw_dset_split_to_augment['text'])
        all_labels = list(raw_dset_split_to_augment['label']) # Should be list of floats

        # For apply_char_truncation_and_adjust_label, it expects original int labels for its logic.
        # So, we need to get the original integer labels for the split to be augmented if we are augmenting.
        # This is a bit tricky as we just converted them to float in full_raw_dset.
        # It's better to load the *original* int labels for augmentation source only when needed.

        # Let's refine: only cast to float for splits NOT being augmented initially.
        # The augmented split will handle its label types carefully.

        # --- REVISED LOGIC --- #
        # Reload full_raw_dset to ensure clean state for this revised logic
        if raw_dset_override is not None:
            full_raw_dset = raw_dset_override # Assume it's fresh for each test call due to function scope fixture
        else:
            full_raw_dset = load_dataset("imdb")
        # --- END REVISED LOGIC --- #

    if enable_augmentation and truncation_methods and apply_to_split in full_raw_dset.keys():
        raw_dset_split_to_augment = full_raw_dset[apply_to_split]
        
        # Texts and labels from the target split for augmentation
        current_split_texts = list(raw_dset_split_to_augment['text'])
        # Original labels from this split (will be cast to float after augmentation or if not augmented)
        current_split_original_labels = list(raw_dset_split_to_augment['label'])

        augmented_texts_collector = []
        augmented_labels_collector = []

        for method in truncation_methods:
            if not method: 
                continue
            for i in range(len(current_split_texts)):
                original_text = current_split_texts[i]
                # Use original integer label for augmentation function's logic
                original_label_for_aug = int(current_split_original_labels[i]) 
                
                truncated_text, adjusted_label = apply_char_truncation_and_adjust_label(
                    original_text,
                    original_label_for_aug, 
                    truncation_percentage,
                    method
                )
                augmented_texts_collector.append(truncated_text)
                augmented_labels_collector.append(adjusted_label) # adjusted_label is already float
        
        # Combine original (cast to float) with augmented (already float)
        all_texts_final = current_split_texts + augmented_texts_collector
        all_labels_final = [float(l) for l in current_split_original_labels] + augmented_labels_collector
        
        expanded_dataset = Dataset.from_dict({'text': all_texts_final, 'label': all_labels_final})
        full_raw_dset[apply_to_split] = expanded_dataset

        # For other splits not augmented, ensure their labels are float
        for split_name in full_raw_dset.keys():
            if split_name != apply_to_split:
                current_split = full_raw_dset[split_name]
                if 'label' in current_split.column_names:
                     full_raw_dset[split_name] = current_split.map(
                         lambda exs: {'label': [float(l) for l in exs['label']]}, 
                         batched=True
                     )
    else:
        # No augmentation enabled, ensure all splits have float labels
        for split_name in full_raw_dset.keys():
            current_split = full_raw_dset[split_name]
            if 'label' in current_split.column_names:
                full_raw_dset[split_name] = current_split.map(
                    lambda exs: {'label': [float(l) for l in exs['label']]}, 
                    batched=True
                )

    # Before tokenization, ensure all label columns are definitively float32 in features
    for split_name_final_cast in full_raw_dset.keys():
        if 'label' in full_raw_dset[split_name_final_cast].column_names:
            # Check current feature type. If it's already float32, this might be redundant but safe.
            # datasets.Value uses string representation like 'float32', 'int64' etc.
            if full_raw_dset[split_name_final_cast].features['label'].dtype != 'float32':
                 full_raw_dset[split_name_final_cast] = full_raw_dset[split_name_final_cast].cast_column(
                    'label', datasets.Value('float32')
                )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=tokenizer_max_length, # Use the renamed parameter
            padding="max_length"
        )

    tokenized = full_raw_dset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=max(1, os.cpu_count() // 2)
    )
    tokenized = tokenized.rename_column("label", "labels")
    # Labels column will now contain floats (either original cast to float, or augmented floats)
    tokenized.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

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

    train_dataloader_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'collate_fn': collator,
        'num_workers': num_workers,
        'pin_memory': True, # pin_memory is fine with num_workers=0
    }
    val_dataloader_params = {
        'batch_size': batch_size,
        'shuffle': False,
        'collate_fn': collator,
        'num_workers': num_workers,
        'pin_memory': True,
    }

    if num_workers > 0:
        train_dataloader_params['persistent_workers'] = True
        train_dataloader_params['prefetch_factor'] = 2
        val_dataloader_params['persistent_workers'] = True
        val_dataloader_params['prefetch_factor'] = 2

    train_dl = DataLoader(dset["train"], **train_dataloader_params)
    val_dl = DataLoader(dset["test"], **val_dataloader_params)
    return train_dl, val_dl
