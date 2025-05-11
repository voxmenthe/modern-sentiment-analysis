import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch

# Import functions to be tested
from src.data_processing import download_and_prepare_datasets, create_dataloaders, add_len

@pytest.fixture
def real_tokenizer_for_test():
    return AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

@pytest.fixture
def sample_raw_dataset():
    '''Fixture for a sample raw dataset (mocking load_dataset output).'''
    data_train = {'text': ['example text 1 for train', 'another example for train', 'third example', 'fourth example'], 'label': [0, 1, 0, 1]}
    data_test = {'text': ['example text 1 for test', 'another example for test'], 'label': [1, 0]}
    train_dataset = Dataset.from_dict(data_train)
    test_dataset = Dataset.from_dict(data_test)
    return DatasetDict({'train': train_dataset, 'test': test_dataset})

@pytest.fixture
def sample_tokenized_dataset(real_tokenizer_for_test, sample_raw_dataset):
    tokenizer = real_tokenizer_for_test
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=10)
    
    tokenized = sample_raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")
    return tokenized

@patch('src.data_processing.load_dataset')
def test_download_and_prepare_datasets(mock_load_dataset, mock_tokenizer_for_download_test, sample_raw_dataset):
    mock_load_dataset.return_value = sample_raw_dataset
    max_len = 128
    tokenized_dset = download_and_prepare_datasets(tokenizer=mock_tokenizer_for_download_test, max_length=max_len)

    mock_load_dataset.assert_called_once_with("imdb")
    # Check that the mock_tokenizer itself was called, which will trigger its side_effect.
    mock_tokenizer_for_download_test.assert_called() 

    assert isinstance(tokenized_dset, DatasetDict)
    assert "train" in tokenized_dset
    assert "test" in tokenized_dset
    assert "text" not in tokenized_dset["train"].column_names
    assert "labels" in tokenized_dset["train"].column_names
    assert "input_ids" in tokenized_dset["train"].column_names
    assert "attention_mask" in tokenized_dset["train"].column_names
    assert tokenized_dset["train"].format["type"] == "torch"
    assert tokenized_dset["test"].format["type"] == "torch"

def test_create_dataloaders(sample_tokenized_dataset, real_tokenizer_for_test):
    batch_size = 4
    train_dl, val_dl = create_dataloaders(dset=sample_tokenized_dataset, tokenizer=real_tokenizer_for_test, batch_size=batch_size)

    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)
    assert train_dl.batch_size == batch_size
    assert val_dl.batch_size == batch_size
    assert isinstance(train_dl.collate_fn, DataCollatorWithPadding)
    assert isinstance(val_dl.collate_fn, DataCollatorWithPadding)
    assert train_dl.collate_fn.tokenizer == real_tokenizer_for_test
    assert val_dl.collate_fn.tokenizer == real_tokenizer_for_test
    
    first_train_batch = next(iter(train_dl))
    assert 'lengths' in first_train_batch
    assert torch.is_tensor(first_train_batch['lengths'])
    assert len(first_train_batch['lengths']) == batch_size 

    assert (first_train_batch['lengths'] > 0).all()

    for i in range(first_train_batch['input_ids'].shape[0]):
        expected_len_in_batch = first_train_batch['attention_mask'][i].sum().long()
        assert first_train_batch['lengths'][i] == expected_len_in_batch
    
    # Check sampler types for shuffle argument verification
    assert isinstance(train_dl.sampler, RandomSampler)
    assert isinstance(val_dl.sampler, SequentialSampler)

    # Check if we can iterate through a batch (this is where pickling often fails)
    try:
        first_train_batch = next(iter(train_dl))
        assert first_train_batch["input_ids"].shape[0] <= batch_size
        # Further checks on batch contents can be added if needed
    except Exception as e:
        pytest.fail(f"Failed to iterate over train_dl with real tokenizer: {e}")

    try:
        first_val_batch = next(iter(val_dl))
        assert first_val_batch["input_ids"].shape[0] <= batch_size
    except Exception as e:
        pytest.fail(f"Failed to iterate over val_dl with real tokenizer: {e}")

@pytest.fixture
def mock_tokenizer_for_download_test():
    mock = MagicMock(spec=AutoTokenizer)
    # Configure mock tokenizer attributes and methods as needed by download_and_prepare_datasets
    mock.pad_token_id = 0
    mock.model_max_length = 512
    # Example of mocking the __call__ behavior if tokenizer is called directly
    def mock_tokenize_call(text, truncation, max_length):
        # Simulate tokenization output structure
        if isinstance(text, list):
            return {
                "input_ids": [[101, 1000, 1002] for _ in text],
                "attention_mask": [[1, 1, 1] for _ in text]
            }
        return {"input_ids": [101, 1000, 1002], "attention_mask": [1,1,1]}
    mock.side_effect = mock_tokenize_call # if tokenizer is called like tokenizer(...)
    # If tokenizer is used like tokenizer.encode_plus or other methods, mock those specifically.
    return mock

def test_add_len_function():
    example_to_process = {"input_ids": torch.tensor([101, 200, 201, 102]), "attention_mask": torch.tensor([1, 1, 1, 0])}
    # Call the add_len function to modify the dictionary
    processed_example = add_len(example_to_process.copy()) # Use .copy() if add_len modifies in-place and you want to preserve original

    assert "lengths" in processed_example
    assert processed_example["lengths"] == 3 # atenção_mask sums to 3 (1+1+1+0)
