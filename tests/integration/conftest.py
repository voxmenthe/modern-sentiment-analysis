import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import ModernBertConfig, AutoTokenizer
from pathlib import Path

from src.models import ModernBertForSentiment
from src.train_utils import SentimentWeightedLoss # Default loss

@pytest.fixture(scope="module")
def tiny_bert_config():
    """Provides a minimal ModernBertConfig for fast testing."""
    config = ModernBertConfig(
        vocab_size=30522, # Standard vocab size, but can be smaller if a dummy vocab is used
        hidden_size=16,    # Very small hidden size
        num_hidden_layers=1, # Minimal layers
        num_attention_heads=1, # Minimal heads
        intermediate_size=32, # Minimal intermediate size
        max_position_embeddings=64, # Small max length
        num_labels=1, # For sentiment logit
        classifier_dropout=0.1,
        output_hidden_states=True, # Set True to allow testing all pooling strategies
        pad_token_id=0,
        # Custom attributes for ModernBertForSentiment
        pooling_strategy='cls', 
        num_weighted_layers=1, # Corresponds to num_hidden_layers for weighted pool
        loss_function={'name': 'SentimentWeightedLoss', 'params': {}}
    )
    return config

@pytest.fixture(scope="module")
def dummy_tokenizer_path_for_integration():
    # Points to the same dummy tokenizer created for data pipeline tests
    base_path = Path(__file__).resolve().parent.parent # 'tests' directory
    return str(base_path / "test_data/dummy_tokenizer")

@pytest.fixture(scope="module")
def integration_test_tokenizer(dummy_tokenizer_path_for_integration):
    """Loads the dummy tokenizer for integration tests."""
    return AutoTokenizer.from_pretrained(dummy_tokenizer_path_for_integration)

@pytest.fixture(scope="function") # Function scope for model to be re-initialized per test
def tiny_modern_bert_sentiment_model(tiny_bert_config):
    """Provides an instance of ModernBertForSentiment with a tiny config."""
    return ModernBertForSentiment(config=tiny_bert_config)

@pytest.fixture(scope="function")
def dummy_integration_dataloader(integration_test_tokenizer):
    """Creates a simple DataLoader with a few tokenized samples for integration tests."""
    texts = ["this is a test", "another test sentence for the model"]
    labels = [1, 0]
    max_length = 16 # Should match or be less than model's max_position_embeddings if relevant
    batch_size = 2

    inputs = integration_test_tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    # Ensure 'labels' are float for BCE-type losses, and long for CrossEntropy. 
    # Our default SentimentWeightedLoss (BCE based) will handle float internally.
    # ModernBertForSentiment itself doesn't cast labels but passes them to loss_fct.
    # The loss_fct (SentimentWeightedLoss) expects float targets.
    label_tensors = torch.tensor(labels, dtype=torch.float)
    lengths = torch.tensor([len(integration_test_tokenizer.encode(t)) for t in texts], dtype=torch.long)

    # Ensure the dataloader provides 'lengths' if the model's loss function needs it
    # The current ModernBertForSentiment forward pass requires 'lengths' if 'labels' are present.
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], label_tensors, lengths)
    
    # Custom collate_fn to return a dictionary matching model's expected input names
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        batched_labels = torch.stack([item[2] for item in batch])
        batched_lengths = torch.stack([item[3] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': batched_labels,
            'lengths': batched_lengths
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

@pytest.fixture(scope="module")
def test_config_for_components(tmp_path_factory):
    # A minimal config dictionary for use in test_training_components.py
    # primarily for functions like load_config or parts of train() that need a config dict.
    output_dir = tmp_path_factory.mktemp("component_test_output")
    return {
        'model': {
            'name': 'dummy_model_for_integration_test', # Not actually loaded from Hub
            'max_length': 16,
            'dropout': 0.1,
            'output_dir': str(output_dir / 'models'),
            'pooling_strategy': 'cls',
            'num_weighted_layers': 1,
            'loss_function': {'name': 'SentimentWeightedLoss', 'params': {}}
        },
        'data': {
            'datasets': ['dummy'], # Not used for these component tests directly
            'local_data_dir': 'dummy_data',
        },
        'training': {
            'batch_size': 2,
            'epochs': 1,
            'lr': 1e-4,
            'weight_decay_rate': 0.01,
            'resume_from_checkpoint': None,
            'eval_steps': 10 # Not critical for most component tests here
        }
    }
