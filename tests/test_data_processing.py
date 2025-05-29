import pytest
import yaml
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import math
import os

from src.data_processing import download_and_prepare_datasets
from src.data_augmentation import apply_char_truncation_and_adjust_label # For direct testing if needed, or reference

# Define a small number of samples for testing to speed things up
N_SAMPLES = 4 # Use a small even number for easier % calculations if needed
TOKENIZER_MAX_LENGTH = 32 # Use a small max length for faster tokenization in tests

@pytest.fixture(scope="function")
def small_imdb_dataset() -> DatasetDict:
    """Fixture to load a small subset of IMDB for testing."""
    # Ensure we have a local copy of imdb or it will download. 
    # For CI, this should ideally be cached or pre-downloaded.
    try:
        train_split = load_dataset("imdb", split=f"train[:{N_SAMPLES}]")
        test_split = load_dataset("imdb", split=f"test[:{N_SAMPLES}]")
    except Exception as e:
        pytest.skip(f"Could not load IMDB dataset (possibly due to network or disk issues): {e}")
    
    # Make copies to avoid modifying the cache / original fixture data across tests if map is in-place
    return DatasetDict({
        "train": Dataset.from_dict(train_split[:]),
        "test": Dataset.from_dict(test_split[:])
    })

@pytest.fixture(scope="session")
def tokenizer_fixture():
    """Fixture for the tokenizer."""
    # Using a basic tokenizer for speed, actual model tokenizer can be used if specific features are tested
    return AutoTokenizer.from_pretrained("bert-base-uncased") 

def create_temp_config(tmp_path_factory, config_data):
    """Helper to create a temporary config.yaml file."""
    config_file = tmp_path_factory.mktemp("config") / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    return str(config_file)

# --- Test Cases --- 

def test_no_augmentation(small_imdb_dataset, tokenizer_fixture, tmp_path_factory):
    """Test dataset processing when augmentation is disabled."""
    config_data = {
        'data': {
            'enable_augmentation': False,
            'truncation_percentage': 0.5, 
            'truncation_method': ["suffix"],
            'apply_to_split': 'train'
        }
    }
    config_path = create_temp_config(tmp_path_factory, config_data)
    
    processed_dset = download_and_prepare_datasets(
        tokenizer=tokenizer_fixture, 
        tokenizer_max_length=TOKENIZER_MAX_LENGTH, 
        config_path=config_path,
        raw_dset_override=small_imdb_dataset
    )

    assert len(processed_dset["train"]) == N_SAMPLES
    assert len(processed_dset["test"]) == N_SAMPLES
    
    # Check that original texts are present (though tokenized)
    # And labels are float
    for split_name in ["train", "test"]:
        original_texts = small_imdb_dataset[split_name]["text"]
        original_labels = small_imdb_dataset[split_name]["label"]
        processed_split = processed_dset[split_name]

        for i in range(N_SAMPLES):
            assert processed_split[i]["labels"].item() == float(original_labels[i])
            assert processed_split[i]["input_ids"].shape[0] == TOKENIZER_MAX_LENGTH
            assert processed_split[i]["attention_mask"].sum().item() <= TOKENIZER_MAX_LENGTH
            # Verify lengths are computed correctly
            assert processed_split[i]["lengths"].item() == processed_split[i]["attention_mask"].sum().item()

        # Ensure all labels in the processed dataset are float
        assert processed_split["labels"].dtype == torch.float32

def test_augmentation_single_method_suffix(small_imdb_dataset, tokenizer_fixture, tmp_path_factory):
    """Test dataset processing with suffix augmentation on the train split."""
    trunc_percentage = 0.5
    config_data = {
        'data': {
            'enable_augmentation': True,
            'truncation_percentage': trunc_percentage,
            'truncation_method': ["suffix"],
            'apply_to_split': 'train'
        }
    }
    config_path = create_temp_config(tmp_path_factory, config_data)
    
    # Deepcopy the fixture because it might be modified by the processing if not careful
    # However, download_and_prepare_datasets should handle this by operating on copies or new dicts.
    # Let's rely on small_imdb_dataset fixture providing fresh copies for each test or being immutable.
    # The current fixture creates new Dataset objects from slices, which should be fine.

    processed_dset = download_and_prepare_datasets(
        tokenizer=tokenizer_fixture, 
        tokenizer_max_length=TOKENIZER_MAX_LENGTH, 
        config_path=config_path,
        raw_dset_override=small_imdb_dataset
    )

    assert len(processed_dset["train"]) == N_SAMPLES * 2 # Original + 1 augmentation method
    assert len(processed_dset["test"]) == N_SAMPLES      # Test split not augmented

    original_train_texts = small_imdb_dataset["train"]["text"]
    original_train_labels = small_imdb_dataset["train"]["label"]

    # Check original samples in train split (first N_SAMPLES)
    for i in range(N_SAMPLES):
        assert processed_dset["train"][i]["labels"].item() == float(original_train_labels[i])
        # Original text tokenized
        reconstructed_original_text_tokenized = tokenizer_fixture.decode(processed_dset["train"][i]["input_ids"], skip_special_tokens=True)
        # This check is tricky due to tokenization differences, focus on label and length for augmented part.

    # Check augmented samples in train split (next N_SAMPLES)
    for i in range(N_SAMPLES):
        original_text = original_train_texts[i]
        original_label = original_train_labels[i]
        augmented_sample_idx = N_SAMPLES + i
        
        # Verify label adjustment
        expected_char_len = math.ceil(len(original_text) * trunc_percentage)
        # Handle cases where truncation might not occur if target_char_len is >= original_char_len
        actual_truncated_text_before_tokenization, expected_label = apply_char_truncation_and_adjust_label(
            original_text, original_label, trunc_percentage, "suffix"
        )
        assert processed_dset["train"][augmented_sample_idx]["labels"].item() == pytest.approx(expected_label)

        # Verify tokenization length for augmented sample
        assert processed_dset["train"][augmented_sample_idx]["input_ids"].shape[0] == TOKENIZER_MAX_LENGTH

    assert processed_dset["train"]["labels"].dtype == torch.float32
    assert processed_dset["test"]["labels"].dtype == torch.float32 # Should also be float

# We can add more tests for prefix, both_ends, multiple methods, and apply_to_split='test'
# For brevity, I'll add one for multiple methods and one for apply_to_split='test'

def test_augmentation_multiple_methods(small_imdb_dataset, tokenizer_fixture, tmp_path_factory):
    """Test with multiple augmentation methods (prefix and suffix)."""
    trunc_percentage = 0.6
    methods = ["prefix", "suffix"]
    config_data = {
        'data': {
            'enable_augmentation': True,
            'truncation_percentage': trunc_percentage,
            'truncation_method': methods,
            'apply_to_split': 'train'
        }
    }
    config_path = create_temp_config(tmp_path_factory, config_data)
    
    processed_dset = download_and_prepare_datasets(
        tokenizer=tokenizer_fixture, 
        tokenizer_max_length=TOKENIZER_MAX_LENGTH, 
        config_path=config_path,
        raw_dset_override=small_imdb_dataset
    )

    assert len(processed_dset["train"]) == N_SAMPLES * (1 + len(methods))
    assert len(processed_dset["test"]) == N_SAMPLES

    original_train_texts = small_imdb_dataset["train"]["text"]
    original_train_labels = small_imdb_dataset["train"]["label"]

    # Check original samples
    for i in range(N_SAMPLES):
        assert processed_dset["train"][i]["labels"].item() == float(original_train_labels[i])

    # Check augmented samples for each method
    for method_idx, method_name in enumerate(methods):
        for i in range(N_SAMPLES):
            original_text = original_train_texts[i]
            original_label = original_train_labels[i]
            augmented_sample_idx = N_SAMPLES * (1 + method_idx) + i
            
            _, expected_label = apply_char_truncation_and_adjust_label(
                original_text, original_label, trunc_percentage, method_name
            )
            assert processed_dset["train"][augmented_sample_idx]["labels"].item() == pytest.approx(expected_label)
            assert processed_dset["train"][augmented_sample_idx]["input_ids"].shape[0] == TOKENIZER_MAX_LENGTH
    
    assert processed_dset["train"]["labels"].dtype == torch.float32

def test_augmentation_apply_to_test_split(small_imdb_dataset, tokenizer_fixture, tmp_path_factory):
    """Test augmentation applied to the 'test' split."""
    trunc_percentage = 0.7
    method = "both_ends"
    config_data = {
        'data': {
            'enable_augmentation': True,
            'truncation_percentage': trunc_percentage,
            'truncation_method': [method],
            'apply_to_split': 'test'
        }
    }
    config_path = create_temp_config(tmp_path_factory, config_data)
    
    processed_dset = download_and_prepare_datasets(
        tokenizer=tokenizer_fixture, 
        tokenizer_max_length=TOKENIZER_MAX_LENGTH, 
        config_path=config_path,
        raw_dset_override=small_imdb_dataset
    )

    assert len(processed_dset["train"]) == N_SAMPLES # Train not augmented
    assert len(processed_dset["test"]) == N_SAMPLES * 2 # Test augmented

    # Check train split (original, labels should be float)
    for i in range(N_SAMPLES):
        assert processed_dset["train"][i]["labels"].item() == float(small_imdb_dataset["train"]["label"][i])
    assert processed_dset["train"]["labels"].dtype == torch.float32

    # Check test split (original + augmented)
    original_test_texts = small_imdb_dataset["test"]["text"]
    original_test_labels = small_imdb_dataset["test"]["label"]

    for i in range(N_SAMPLES):
        assert processed_dset["test"][i]["labels"].item() == float(original_test_labels[i])
    
    for i in range(N_SAMPLES):
        original_text = original_test_texts[i]
        original_label = original_test_labels[i]
        augmented_sample_idx = N_SAMPLES + i
        
        _, expected_label = apply_char_truncation_and_adjust_label(
            original_text, original_label, trunc_percentage, method
        )
        assert processed_dset["test"][augmented_sample_idx]["labels"].item() == pytest.approx(expected_label)
        assert processed_dset["test"][augmented_sample_idx]["input_ids"].shape[0] == TOKENIZER_MAX_LENGTH

    assert processed_dset["test"]["labels"].dtype == torch.float32

def test_augmentation_empty_method_list(small_imdb_dataset, tokenizer_fixture, tmp_path_factory):
    """Test that an empty truncation_method list results in no augmentation, even if enabled."""
    config_data = {
        'data': {
            'enable_augmentation': True,
            'truncation_percentage': 0.5,
            'truncation_method': [], # Empty list
            'apply_to_split': 'train'
        }
    }
    config_path = create_temp_config(tmp_path_factory, config_data)
    
    processed_dset = download_and_prepare_datasets(
        tokenizer=tokenizer_fixture, 
        tokenizer_max_length=TOKENIZER_MAX_LENGTH, 
        config_path=config_path,
        raw_dset_override=small_imdb_dataset
    )

    assert len(processed_dset["train"]) == N_SAMPLES
    assert len(processed_dset["test"]) == N_SAMPLES
    assert processed_dset["train"]["labels"].dtype == torch.float32
    assert processed_dset["test"]["labels"].dtype == torch.float32

def test_truncation_does_not_occur_if_percentage_too_high(small_imdb_dataset, tokenizer_fixture, tmp_path_factory):
    """Test that if truncation_percentage results in target_len >= original_len, no truncation occurs and label is original."""
    trunc_percentage = 1.5 # 150%, so no truncation should occur
    config_data = {
        'data': {
            'enable_augmentation': True,
            'truncation_percentage': trunc_percentage,
            'truncation_method': ["suffix"],
            'apply_to_split': 'train'
        }
    }
    config_path = create_temp_config(tmp_path_factory, config_data)
    
    processed_dset = download_and_prepare_datasets(
        tokenizer=tokenizer_fixture, 
        tokenizer_max_length=TOKENIZER_MAX_LENGTH, 
        config_path=config_path,
        raw_dset_override=small_imdb_dataset
    )

    # Even though augmentation is "enabled", no actual change to text/label for augmentation part
    assert len(processed_dset["train"]) == N_SAMPLES * 2
    original_train_labels = small_imdb_dataset["train"]["label"]

    # Check "augmented" samples (which should be identical to original due to high percentage)
    for i in range(N_SAMPLES):
        augmented_sample_idx = N_SAMPLES + i
        # Label should be original label, cast to float
        assert processed_dset["train"][augmented_sample_idx]["labels"].item() == float(original_train_labels[i])

# No new content, just ensuring the file ends correctly if there was an unexpected character or truncation. 