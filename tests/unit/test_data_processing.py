import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch

# Import functions to be tested
from src.data_processing import download_and_prepare_datasets, create_dataloaders

@pytest.fixture
def mock_tokenizer():
    '''Fixture for a mock tokenizer with a .pad() method for DataCollatorWithPadding.'''
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.pad_token_id = 0
    tokenizer.model_max_length = 512
    tokenizer.padding_side = "right"

    def mock_tokenize_fn(text_or_batch, truncation=True, max_length=256, padding=False, add_special_tokens=True, **kwargs):
        if isinstance(text_or_batch, str):
            return {'input_ids': [101, 200, 102], 'attention_mask': [1, 1, 1]}
        output_batch = {'input_ids': [], 'attention_mask': []}
        for _ in text_or_batch:
            output_batch['input_ids'].append([101, 200, 201, 102])
            output_batch['attention_mask'].append([1, 1, 1, 1])
        return output_batch
    
    # Set side_effect for the tokenizer mock itself to execute mock_tokenize_fn when called.
    tokenizer.side_effect = mock_tokenize_fn

    def mock_pad_fn(encoded_inputs, padding=True, max_length=None, return_tensors="pt", **kwargs):
        if not encoded_inputs:
            return {}
        processed_features = []
        current_max_length_in_batch = 0
        # Ensure keys_in_batch is robust if encoded_inputs[0] is not fully populated or is an empty dict
        keys_in_batch = list(encoded_inputs[0].keys()) if encoded_inputs and encoded_inputs[0] else []

        for item in encoded_inputs:
            processed_item = {}
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    if value.ndim == 0:
                        processed_item[key] = [value.item()]
                    else:
                        processed_item[key] = value.tolist()
                elif not isinstance(value, list):
                     processed_item[key] = [value]
                else:
                    processed_item[key] = value
                
                if key == 'input_ids':
                    current_max_length_in_batch = max(current_max_length_in_batch, len(processed_item[key]))
            processed_features.append(processed_item)

        effective_max_length = max_length if max_length is not None else current_max_length_in_batch
        final_batch_dict = {key: [] for key in keys_in_batch}

        for item in processed_features:
            for key in keys_in_batch:
                current_val_list = item[key]
                if key in ['input_ids', 'attention_mask']:
                    pad_value = tokenizer.pad_token_id if key == 'input_ids' else 0
                    pad_len = effective_max_length - len(current_val_list)
                    padded_list = current_val_list + [pad_value] * pad_len
                    final_batch_dict[key].append(padded_list[:effective_max_length])
                else: 
                    final_batch_dict[key].append(current_val_list[0] if isinstance(current_val_list, list) and len(current_val_list) == 1 and not isinstance(current_val_list[0], list) else current_val_list)
        
        if return_tensors == "pt":
            for key, value_lists in final_batch_dict.items():
                try:
                    if key in ['input_ids', 'attention_mask', 'labels', 'lengths']:
                         final_batch_dict[key] = torch.tensor(value_lists, dtype=torch.long)
                except Exception as e:
                    print(f"Warning: Could not convert key '{key}' to stacked tensor in mock_pad_fn: {e}")
                    pass 
            return final_batch_dict
        else:
            raise NotImplementedError("Mock only supports return_tensors='pt' for pad method")

    tokenizer.pad = MagicMock(side_effect=mock_pad_fn)
    return tokenizer

@pytest.fixture
def sample_raw_dataset():
    '''Fixture for a sample raw dataset (mocking load_dataset output).'''
    data_train = {'text': ['example text 1 for train', 'another example for train', 'third example', 'fourth example'], 'label': [0, 1, 0, 1]}
    data_test = {'text': ['example text 1 for test', 'another example for test'], 'label': [1, 0]}
    train_dataset = Dataset.from_dict(data_train)
    test_dataset = Dataset.from_dict(data_test)
    return DatasetDict({'train': train_dataset, 'test': test_dataset})

@pytest.fixture
def sample_tokenized_dataset():
    '''Fixture for a sample tokenized dataset. Now with 4 train samples.'''
    train_data = {
        'input_ids': torch.tensor([
            [101, 2054, 2022, 102], [101, 2064, 2022, 102],
            [101, 2000, 2001, 102], [101, 2002, 2003, 102]
        ]),
        'attention_mask': torch.tensor([
            [1, 1, 1, 1], [1, 1, 1, 1],
            [1, 1, 1, 1], [1, 1, 1, 1]
        ]),
        'labels': torch.tensor([0, 1, 0, 1])
    }
    test_data = {
        'input_ids': torch.tensor([[101, 2054, 2023, 102], [101, 2064, 2023, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
        'labels': torch.tensor([1, 0])
    }
    train_dataset = Dataset.from_dict(train_data).with_format("torch")
    test_dataset = Dataset.from_dict(test_data).with_format("torch")
    return DatasetDict({'train': train_dataset, 'test': test_dataset})


@patch('src.data_processing.load_dataset')
def test_download_and_prepare_datasets(mock_load_dataset, mock_tokenizer, sample_raw_dataset):
    mock_load_dataset.return_value = sample_raw_dataset
    max_len = 128
    tokenized_dset = download_and_prepare_datasets(tokenizer=mock_tokenizer, max_length=max_len)

    mock_load_dataset.assert_called_once_with("imdb")
    # Check that the mock_tokenizer itself was called, which will trigger its side_effect.
    mock_tokenizer.assert_called() 

    assert isinstance(tokenized_dset, DatasetDict)
    assert "train" in tokenized_dset
    assert "test" in tokenized_dset
    assert "text" not in tokenized_dset["train"].column_names
    assert "labels" in tokenized_dset["train"].column_names
    assert "input_ids" in tokenized_dset["train"].column_names
    assert "attention_mask" in tokenized_dset["train"].column_names
    assert tokenized_dset["train"].format["type"] == "torch"
    assert tokenized_dset["test"].format["type"] == "torch"

def test_create_dataloaders(sample_tokenized_dataset, mock_tokenizer):
    batch_size = 4
    train_dl, val_dl = create_dataloaders(dset=sample_tokenized_dataset, tokenizer=mock_tokenizer, batch_size=batch_size)

    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)
    assert train_dl.batch_size == batch_size
    assert val_dl.batch_size == batch_size
    assert isinstance(train_dl.collate_fn, DataCollatorWithPadding)
    assert isinstance(val_dl.collate_fn, DataCollatorWithPadding)
    assert train_dl.collate_fn.tokenizer == mock_tokenizer
    assert val_dl.collate_fn.tokenizer == mock_tokenizer
    
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
