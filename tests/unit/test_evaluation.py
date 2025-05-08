import pytest
import torch
from unittest.mock import MagicMock, patch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef

from src.evaluation import evaluate

@pytest.fixture
def mock_model_output_fixture():
    '''Fixture to create a mock model output object.'''
    def _create_output(loss_val, logits_tensor):
        output = MagicMock()
        output.loss = MagicMock(spec=torch.Tensor)
        output.loss.item = MagicMock(return_value=loss_val)
        output.logits = logits_tensor
        return output
    return _create_output

@pytest.fixture
def mock_model_fixture(mock_model_output_fixture):
    '''Fixture for a mock model.'''
    model = MagicMock()
    model.eval = MagicMock()
    return model

@pytest.fixture
def mock_batch_fixture():
    '''Fixture to create a single mock batch with real tensors.'''
    def _create_batch(labels_data, include_lengths=False, batch_size=4, seq_len=10):
        labels = torch.tensor(labels_data)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        if include_lengths:
            batch['lengths'] = torch.randint(1, seq_len, (batch_size,))
        return batch
    return _create_batch

@pytest.fixture
def mock_dataloader_fixture():
    '''Fixture for a mock dataloader that yields prepared batches.'''
    def _create_loader(list_of_batches):
        dataloader_mock = MagicMock()
        dataloader_mock.__iter__.return_value = iter(list_of_batches)
        dataloader_mock.__len__.return_value = len(list_of_batches) if list_of_batches else 1
        return dataloader_mock
    return _create_loader
