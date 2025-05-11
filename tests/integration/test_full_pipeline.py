import sys # Keep for other potential uses, though path mod is now in conftest
import os

# --- BEGIN WORKTREE PATH FIX ---
# Get the absolute path to the current project's root (mps_debottleneck_training)
# Assuming tests/integration/test_full_pipeline.py is in WORKSPACE_ROOT/tests/integration/
current_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Prepend the worktree's project root to sys.path
if current_project_root not in sys.path:
     sys.path.insert(0, current_project_root)
# print(f"DEBUG: Modified sys.path for worktree: {sys.path}") # Optional: can be re-enabled if needed
# --- END WORKTREE PATH FIX ---

print(f"DEBUG: TOP OF TEST FILE: Executing tests/integration/test_full_pipeline.py from: {os.path.abspath(__file__)}") # Kept for verification
print(f"DEBUG: TOP OF TEST FILE: sys.path (at import time of this file): {sys.path}") # Kept for verification

import pytest
import yaml
from pathlib import Path
import shutil
import torch
from unittest.mock import patch, MagicMock
import json # Added import

# Function to test
from src import train as train_module # Import train module to check its path
from src.data_processing import download_and_prepare_datasets # For path reference in patching
from datasets import Dataset, DatasetDict # For constructing mock return value
from transformers import ModernBertConfig, ModernBertModel # Added ModernBertModel
from tests.integration.conftest import tiny_bert_config # Import the fixture function

# Helper to parse the dummy data files (similar to test_data_pipeline.py)
def parse_dummy_data_for_full_pipeline(file_path):
    texts = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' label=')
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(int(parts[1]))
    return {'text': texts, 'label': labels}

@pytest.fixture(scope="function")
def full_pipeline_config_path(tmp_path):
    test_output_dir = tmp_path / "test_run_outputs"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    base_test_dir = Path(__file__).resolve().parent
    original_config_file = base_test_dir / "test_config_full_pipeline.yaml"
    with open(original_config_file, 'r') as f:
        config = yaml.safe_load(f)
    config['model']['output_dir'] = str(test_output_dir / "model_outputs")
    config['logging']['log_file'] = str(test_output_dir / "train.log")
    if 'visualization_dir' in config.get('training', {}):
        config['training']['visualization_dir'] = str(test_output_dir / "visualizations")
    temp_config_file = tmp_path / "test_config_modified.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(config, f)
    return temp_config_file

class TestFullTrainingPipeline:
    @patch('src.train.download_and_prepare_datasets')
    @patch('src.train.ModernBertModel.from_pretrained')
    @patch('src.train.ModernBertConfig.from_pretrained')
    def test_train_function_completes_with_dummy_data(
        self, 
        mock_config_from_pretrained, 
        mock_model_from_pretrained, 
        mock_download_and_prepare_datasets, 
        full_pipeline_config_path,
        tiny_bert_config
    ):
        print("DEBUG: Starting test_train_function_completes_with_dummy_data") # Kept
        config = train_module.load_config(str(full_pipeline_config_path))
        dummy_tokenizer_base_path = Path(__file__).resolve().parent.parent
        absolute_dummy_tokenizer_path = str(dummy_tokenizer_base_path / "test_data/dummy_tokenizer")
        config['model']['name'] = absolute_dummy_tokenizer_path
        mock_prepared_bert_config = tiny_bert_config
        mock_prepared_bert_config.classifier_dropout = config['model']['dropout']
        mock_prepared_bert_config.pooling_strategy = config['model'].get('pooling_strategy', tiny_bert_config.pooling_strategy)
        mock_prepared_bert_config.num_weighted_layers = config['model'].get('num_weighted_layers', tiny_bert_config.num_weighted_layers)
        mock_prepared_bert_config.loss_function = config['model'].get('loss_function', tiny_bert_config.loss_function)
        if mock_prepared_bert_config.pooling_strategy in ['weighted_layer', 'cls_weighted_concat']:
            mock_prepared_bert_config.output_hidden_states = True
        else:
            mock_prepared_bert_config.output_hidden_states = False
        mock_config_from_pretrained.return_value = mock_prepared_bert_config
        mock_model_instance = ModernBertModel(config=mock_prepared_bert_config)
        mock_model_from_pretrained.return_value = mock_model_instance
        dummy_data_base_path = Path(__file__).resolve().parent.parent / "test_data/dummy_dataset"
        train_data_content = parse_dummy_data_for_full_pipeline(dummy_data_base_path / "train/data.txt")
        val_data_content = parse_dummy_data_for_full_pipeline(dummy_data_base_path / "validation/data.txt")
        raw_dsets = DatasetDict({
            "train": Dataset.from_dict(train_data_content),
            "test": Dataset.from_dict(val_data_content)
        })
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        max_len = config['model']['max_length']
        def tokenize_func(examples):
            tokenized_batch = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_len)
            tokenized_batch['labels'] = [float(l) for l in examples['label']]
            return tokenized_batch
        tokenized_dsets = raw_dsets.map(tokenize_func, batched=True, remove_columns=['text', 'label'])
        tokenized_dsets.set_format("torch")
        mock_download_and_prepare_datasets.return_value = tokenized_dsets
        print(f"DEBUG: train_module.__file__ is: {train_module.__file__}") # Kept
        try:
            train_module.train(config)
            print("DEBUG: train(config) completed in test.") # Kept
        except Exception as e:
            pytest.fail(f"train() function failed during full pipeline test: {e}\nOutput dir: {config['model']['output_dir']}")
        mock_download_and_prepare_datasets.assert_called_once()
        mock_config_from_pretrained.assert_called_once_with(config['model']['name'])
        mock_model_from_pretrained.assert_called_once_with(config['model']['name'], config=mock_prepared_bert_config)
        output_dir = Path(config['model']['output_dir'])
        log_file = Path(config['logging']['log_file'])
        log_dir = log_file.parent
        assert log_dir.exists(), f"Log directory not found: {log_dir}"
        assert output_dir.exists(), f"Model output directory not found: {output_dir}"
        checkpoint_files = list(output_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0, f"No checkpoint .pt files found in {output_dir}"
        metrics_files = list(output_dir.glob("*metrics.json"))
        assert len(metrics_files) == 1, f"Expected 1 metrics JSON file, found {len(metrics_files)} in {output_dir}"
        metrics_file_path = metrics_files[0]
        with open(metrics_file_path, 'r') as f:
            history = json.load(f)
            print(f"DEBUG: Loaded history object in test: {history}") # Kept
        assert isinstance(history, dict)
        assert "train_loss" in history
        assert isinstance(history["train_loss"], list)
        assert len(history["train_loss"]) == config['training']['epochs']
        if config['training']['epochs'] > 0:
            assert isinstance(history["train_loss"][0], float)
        detailed_train_metrics_to_check_empty = [
            "train_accuracy", "train_f1", "train_roc_auc", 
            "train_precision", "train_recall", "train_mcc"
        ]
        for metric_key in detailed_train_metrics_to_check_empty:
            assert metric_key in history, f"Key {metric_key} expected in history (as empty list)"
            assert isinstance(history[metric_key], list), f"{metric_key} should be a list"
            assert len(history[metric_key]) == 0, f"{metric_key} should be an empty list, got {history[metric_key]}"
        assert "epoch" in history
        assert isinstance(history["epoch"], list)
        expected_epochs = config['training']['epochs']
        actual_epochs = len(history["epoch"])
        epoch_msg = f"Expected {expected_epochs} epoch entries, got {actual_epochs}"
        assert actual_epochs == expected_epochs, epoch_msg
        expected_val_metrics = ["val_loss", "val_accuracy", "val_f1", "val_roc_auc", "val_precision", "val_recall", "val_mcc"]
        for val_metric_key in expected_val_metrics:
            assert val_metric_key in history, f"Validation metric key {val_metric_key} missing from history"
            assert isinstance(history[val_metric_key], list)
            actual_len = len(history[val_metric_key])
            val_metric_msg = f"Expected {expected_epochs} entries for {val_metric_key}, got {actual_len}"
            assert actual_len == expected_epochs, val_metric_msg
