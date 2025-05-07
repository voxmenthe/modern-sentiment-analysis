import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from pathlib import Path

# Assuming these are the correct import paths for the functions to test
# If create_dataloaders is in src.train, we'd import from there.
# Let's assume it's accessible from data_processing or a utils file.
# For now, using the import from train.py where it's defined.
from src.train import create_dataloaders # Or from src.data_processing if moved

# Helper to parse the dummy data files
def parse_dummy_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' label=')
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(int(parts[1]))
    return {'text': texts, 'label': labels}

@pytest.fixture(scope="module")
def dummy_tokenizer_path():
    base_path = Path(__file__).resolve().parent.parent # 'tests' directory
    return str(base_path / "test_data/dummy_tokenizer")

@pytest.fixture(scope="module")
def raw_dummy_datasets(dummy_dataset_paths):
    train_data = parse_dummy_data(dummy_dataset_paths["train"])
    val_data = parse_dummy_data(dummy_dataset_paths["validation"])
    return DatasetDict({
        "train": Dataset.from_dict(train_data),
        "test": Dataset.from_dict(val_data)
    })

@pytest.fixture(scope="module")
def dummy_dataset_paths():
    base_path = Path(__file__).resolve().parent.parent # 'tests' directory
    train_path = base_path / "test_data/dummy_dataset/train/data.txt"
    val_path = base_path / "test_data/dummy_dataset/validation/data.txt"
    # Make sure these paths are correct relative to the test file execution dir
    return {"train": str(train_path), "validation": str(val_path)}


class TestDataPipelineIntegration:

    def test_tokenization_and_dataloader_creation(self, raw_dummy_datasets, dummy_tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(dummy_tokenizer_path)
        max_length = 32 # Small max_length for testing
        batch_size = 2

        # 1. Manually perform the tokenization step (similar to what download_and_prepare_datasets would do)
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

        tokenized_datasets = raw_dummy_datasets.map(tokenize_function, batched=True)
        
        # Rename 'label' to 'labels' as expected by Hugging Face trainers/models
        # and set format for PyTorch
        # Note: The ModernBertForSentiment model in src/models.py seems to expect 'labels' for loss calculation.
        # The create_dataloaders function in src/train.py also prepares 'labels'.
        if 'label' in tokenized_datasets['train'].column_names:
             tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        
        # Ensure all necessary columns are present for set_format
        # 'labels' is already int. input_ids, attention_mask are created by tokenizer.
        columns_to_set_format = ['input_ids', 'attention_mask']
        if 'labels' in tokenized_datasets['train'].column_names:
            columns_to_set_format.append('labels')
        
        tokenized_datasets.set_format(type='torch', columns=columns_to_set_format)

        # 2. Test create_dataloaders
        # The create_dataloaders function from train.py takes tokenizer as an argument for collate_fn, but it's not strictly used by default collator if data is pre-tokenized.
        # However, it might use it for DataCollatorWithPadding. Let's pass it.
        train_dl, val_dl = create_dataloaders(tokenized_datasets, tokenizer, batch_size)

        assert isinstance(train_dl, DataLoader)
        assert isinstance(val_dl, DataLoader)
        assert train_dl.batch_size == batch_size
        assert val_dl.batch_size == batch_size

        # Check content of a training batch
        train_batch = next(iter(train_dl))
        assert 'input_ids' in train_batch
        assert 'attention_mask' in train_batch
        if 'labels' in columns_to_set_format:
            assert 'labels' in train_batch
            assert train_batch['labels'].shape == (batch_size,)
            # The create_dataloaders in train.py converts labels to long for CrossEntropyLoss typically.
            # Our dummy labels are int, which is fine. If BCEWithLogitsLoss is used, they become float later.
            # The loss functions in train_utils.py expect float targets and handle .view(-1).
            # ModernBertForSentiment's forward pass expects labels to be long if passed to CrossEntropyLoss type losses.
            # For BCE type, it's often float. Let's check based on what create_dataloaders produces.
            # create_dataloaders in train.py does: sample["labels"] = torch.tensor(sample["labels"], dtype=torch.long)
            assert train_batch['labels'].dtype == torch.int64 

        assert train_batch['input_ids'].shape == (batch_size, max_length)
        assert train_batch['input_ids'].dtype == torch.int64
        assert train_batch['attention_mask'].dtype == torch.int64 # Usually long from tokenizer output

        # Check content of a validation batch (handle smaller last batch)
        num_val_samples = len(raw_dummy_datasets['test'])
        expected_val_batch_size = min(batch_size, num_val_samples)
        if expected_val_batch_size > 0:
            val_batch = next(iter(val_dl))
            assert 'input_ids' in val_batch
            if 'labels' in columns_to_set_format:
                assert 'labels' in val_batch
                assert val_batch['labels'].shape == (expected_val_batch_size,)
        else:
            assert len(val_dl) == 0
