import pytest
import torch
import yaml
import os
import re
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call, ANY
import copy
import json

# Functions and classes to test
from src.train import load_config, train 
# Assuming other necessary imports from src like models, data_processing, evaluation are handled by mocks

# Default config for many tests
DEFAULT_TEST_CONFIG = {
    'model': {
        'name': 'prajjwal1/bert-tiny',
        'max_length': 128,
        'dropout': 0.1,
        'num_labels': 1,
        'output_dir': 'test_output/models',
        'pooling_strategy': 'cls',
        'num_weighted_layers': 4,
        'loss_function': {'name': 'SentimentWeightedLoss', 'params': {}}
    },
    'data': {
        'datasets': ['imdb'],
        'local_data_dir': 'test_data/sentiment_data',
    },
    'training': {
        'batch_size': 2,
        'epochs': 1,
        'lr': 1e-5,
        'weight_decay_rate': 0.01,
        'resume_from_checkpoint': None,
        'eval_steps': 10
    }
}

@pytest.fixture
def temp_config_file(tmp_path):
    config_content = copy.deepcopy(DEFAULT_TEST_CONFIG)
    p = tmp_path / "test_config.yaml"
    with open(p, 'w') as f:
        yaml.dump(config_content, f)
    return str(p)

@pytest.fixture
def mock_config():
    # Provide a fresh copy of the default test config for mocks
    return copy.deepcopy(DEFAULT_TEST_CONFIG)

@pytest.fixture
def mock_dependencies(mock_config: dict, tmp_path: Path):
    with patch('src.train.AutoTokenizer.from_pretrained') as mock_tokenizer,\
         patch('src.train.download_and_prepare_datasets') as mock_download_data,\
         patch('src.train.create_dataloaders') as mock_create_dataloaders,\
         patch('src.train.ModernBertConfig.from_pretrained') as mock_bert_config_load,\
         patch('src.train.ModernBertModel.from_pretrained') as mock_bert_model_load,\
         patch('src.train.ModernBertForSentiment') as mock_custom_model,\
         patch('src.train.AdamW') as mock_adamw,\
         patch('src.train.LinearLR') as mock_lr_scheduler,\
         patch('src.train.evaluate') as mock_evaluate,\
         patch('torch.save') as mock_torch_save,\
         patch('os.makedirs') as mock_os_makedirs,\
         patch('os.path.exists') as mock_os_path_exists,\
         patch('torch.load') as mock_torch_load,\
         patch('src.train.re.search') as mock_re_search,\
         patch('torch.device') as mock_torch_device,\
         patch('src.train.generate_artifact_name') as mock_generate_artifact_name:

        mock_tokenizer.return_value = MagicMock(name='tokenizer')
        mock_download_data.return_value = (MagicMock(name='train_texts'), MagicMock(name='val_texts'))
        
        mock_train_dl = MagicMock(name='train_dl')
        mock_val_dl = MagicMock(name='val_dl')
        type(mock_train_dl).__len__ = lambda x: 5 
        type(mock_val_dl).__len__ = lambda x: 2 
        # Make train_dl iterable with at least one batch
        mock_batch = {
            'input_ids': torch.randint(0, 100, (2, 10)), 
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 1, (2,1))
        }
        mock_train_dl.__iter__.return_value = [mock_batch] 
        mock_create_dataloaders.return_value = (mock_train_dl, mock_val_dl)

        mock_config_instance = MagicMock(name='config_instance')
        mock_config_instance.hidden_size = 768
        mock_config_instance.num_labels = mock_config['model']['num_labels'] 
        mock_config_instance.pooling_strategy = mock_config['model']['pooling_strategy']
        mock_config_instance.classifier_dropout = mock_config['model']['dropout']
        mock_bert_config_load.return_value = mock_config_instance

        mock_base_bert_model_instance = MagicMock(name='base_bert_model')
        mock_bert_model_load.return_value = mock_base_bert_model_instance

        mock_custom_model_instance = MagicMock(name='custom_model_instance')
        mock_output = MagicMock(loss=torch.tensor(0.5, requires_grad=True))
        mock_custom_model_instance.return_value = mock_output 
        mock_custom_model_instance.config = mock_config_instance 
        mock_custom_model.return_value = mock_custom_model_instance

        mock_optimizer_instance = MagicMock(name='optimizer')
        mock_adamw.return_value = mock_optimizer_instance

        mock_scheduler_instance = MagicMock(name='scheduler')
        mock_lr_scheduler.return_value = mock_scheduler_instance

        default_metrics = {
            'loss': 0.1, 'accuracy': 0.9, 'f1': 0.85, 
            'roc_auc': 0.92, 'precision': 0.88, 'recall': 0.82, 'mcc': 0.75
        }
        mock_evaluate.return_value = default_metrics

        mock_os_path_exists.return_value = False 
        mock_torch_load.return_value = {
            'model_state_dict': MagicMock(), 'optimizer_state_dict': MagicMock(),
            'scheduler_state_dict': MagicMock(), 'epoch': 0, 'best_f1': 0.0, 'config': mock_config
        }
        mock_match_object = MagicMock()
        mock_match_object.groupdict.return_value = {'epoch': '0', 'f1': '0.0'}
        mock_re_search.return_value = mock_match_object
        mock_torch_device.return_value = 'cpu'

        # Mock generate_artifact_name to return a predictable path
        def mock_generate_artifact_name_impl(*args, **kwargs):
            base_dir = args[0] if args else kwargs.get('base_output_dir', 'test_output/models')
            model_config_name_arg = args[1] if len(args) > 1 else kwargs.get('model_config_name', 'bert-tiny')
            simple_model_name = model_config_name_arg.split('/')[-1] # Correctly get simple model name
            loss_acronym = 'SWL'  # Default to SWL for tests
            epoch = args[3] if len(args) > 3 else kwargs.get('epoch', 1)
            artifact_type = args[4] if len(args) > 4 else kwargs.get('artifact_type', 'metrics')
            timestamp_str = kwargs.get('timestamp_str', '20250509215244') # Use provided or default timestamp
            extension = kwargs.get('extension', 'json')
            f1_score = kwargs.get('f1_score')

            filename_parts = [simple_model_name, timestamp_str, loss_acronym, f"e{epoch}"]
            if artifact_type in ["checkpoint", "plot_confusion_matrix"] and f1_score is not None:
                filename_parts.append(f"{f1_score:.4f}f1")
            elif artifact_type in ["checkpoint", "plot_confusion_matrix"] and f1_score is None:
                 filename_parts.append("NOF1") # Match real function's behavior if f1 is None for these types
            
            filename_core = "_".join(filename_parts)
            suffix_parts = []
            if artifact_type == "checkpoint":
                suffix_parts.append("checkpoint")
            elif artifact_type == "metrics":
                suffix_parts.append("metrics")
            # Add other artifact types if necessary, mirroring the real function
            else:
                suffix_parts.append(artifact_type)
            
            final_filename_str = f"{filename_core}_{'_'.join(suffix_parts)}" if suffix_parts else filename_core
            filename_with_ext = f"{final_filename_str}.{extension.lstrip('.')}"

            return Path(base_dir) / filename_with_ext
        
        mock_generate_artifact_name.side_effect = mock_generate_artifact_name_impl

        yield {
            'mock_tokenizer': mock_tokenizer,
            'mock_download_data': mock_download_data,
            'mock_create_dataloaders': mock_create_dataloaders,
            'mock_bert_config_load': mock_bert_config_load,
            'mock_bert_config_instance': mock_config_instance, 
            'mock_bert_model_load': mock_bert_model_load,
            'mock_custom_model': mock_custom_model, 
            'mock_model_instance': mock_custom_model_instance, 
            'mock_adamw': mock_adamw, 
            'mock_optimizer_instance': mock_optimizer_instance, 
            'mock_lr_scheduler': mock_lr_scheduler, 
            'mock_scheduler_instance': mock_scheduler_instance, 
            'mock_evaluate': mock_evaluate,
            'mock_torch_save': mock_torch_save,
            'mock_os_makedirs': mock_os_makedirs,
            'mock_os_path_exists': mock_os_path_exists,
            'mock_torch_load': mock_torch_load,
            'mock_re_search': mock_re_search,
            'mock_torch_device': mock_torch_device,
            'mock_base_bert_model_instance': mock_base_bert_model_instance,
            'mock_generate_artifact_name': mock_generate_artifact_name,
        }

class TestLoadConfig:
    def test_load_config_success(self, temp_config_file):
        config = load_config(temp_config_file)
        assert config['model']['name'] == 'prajjwal1/bert-tiny'
        assert config['training']['epochs'] == 1

    def test_load_config_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_config.yaml")

    @patch("builtins.open", new_callable=mock_open, read_data="invalid_yaml_content:")
    @patch("yaml.safe_load")
    def test_load_config_yaml_error(self, mock_safe_load, mock_file_open):
        mock_safe_load.side_effect = yaml.YAMLError("Failed to parse")
        with pytest.raises(yaml.YAMLError):
            load_config("any_path.yaml")

class TestTrainFunction:

    def test_device_selection(self, mock_dependencies, temp_config_file):
        config = load_config(temp_config_file)

        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            train(config.copy()) 
            mock_dependencies['mock_torch_device'].assert_called_with('mps')
        
        mock_dependencies['mock_torch_device'].reset_mock()
        with patch('torch.cuda.is_available', return_value=True):
            train(config.copy())
            mock_dependencies['mock_torch_device'].assert_called_with('cuda')

        mock_dependencies['mock_torch_device'].reset_mock()
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            train(config.copy())
            mock_dependencies['mock_torch_device'].assert_called_with('cpu')

    def test_basic_training_flow_no_checkpoint(self, mock_dependencies, temp_config_file, tmp_path):
        config = load_config(temp_config_file)
        output_dir = tmp_path / "test_output_basic" / "models"
        config['model']['output_dir'] = str(output_dir)
        config['training']['epochs'] = 1

        mock_evaluate = mock_dependencies['mock_evaluate']
        mock_torch_save = mock_dependencies['mock_torch_save']
        mock_os_path_exists = mock_dependencies['mock_os_path_exists']
        mock_generate_artifact_name = mock_dependencies['mock_generate_artifact_name']

        mock_os_path_exists.return_value = False
        mock_evaluate.return_value = {
            'loss': 0.1, 'accuracy': 0.9, 'f1': 0.85,
            'roc_auc': 0.92, 'precision': 0.88, 'recall': 0.82, 'mcc': 0.75
        }

        train(config)

        mock_dependencies['mock_tokenizer'].assert_called_once()
        mock_dependencies['mock_download_data'].assert_called_once()
        mock_dependencies['mock_create_dataloaders'].assert_called_once()
        mock_dependencies['mock_bert_config_load'].assert_called_once_with(config['model']['name'])
        assert mock_dependencies['mock_bert_config_instance'].num_labels == config['model']['num_labels']
        mock_dependencies['mock_custom_model'].assert_called_once_with(config=mock_dependencies['mock_bert_config_instance'])
        mock_dependencies['mock_model_instance'].bert = mock_dependencies['mock_base_bert_model_instance']
        mock_dependencies['mock_model_instance'].to.assert_called_once_with(ANY)
        mock_dependencies['mock_optimizer_instance'].step.assert_called_once()
        mock_dependencies['mock_scheduler_instance'].step.assert_called_once()
        mock_evaluate.assert_called() # Called for train and val
        assert mock_evaluate.call_count == 2 # 1 epoch * 2 (train + val)
        mock_torch_save.assert_called_once() # Checkpoint save

        metrics_file_call_args = None
        for call in mock_generate_artifact_name.call_args_list:
            if call.kwargs.get('artifact_type') == 'metrics':
                metrics_file_call_args = call
                break
        assert metrics_file_call_args is not None, "generate_artifact_name not called for metrics"
        
        # Get the actual path generated by the mock for the metrics file
        actual_metrics_path = mock_generate_artifact_name.side_effect(*metrics_file_call_args.args, **metrics_file_call_args.kwargs)
        
        assert Path(config['model']['output_dir']).exists(), f"Base output directory {config['model']['output_dir']} was not created."
        assert Path(actual_metrics_path).exists(), f"Metrics file {actual_metrics_path} was not created."
        assert Path(actual_metrics_path).parent.exists(), f"Metrics file directory {Path(actual_metrics_path).parent} was not created."

    def test_bert_config_modifications_weighted_pooling(self, mock_dependencies, temp_config_file):
        config = load_config(temp_config_file)
        config['model']['output_dir'] = str(Path(temp_config_file).parent)
        config['model']['pooling_strategy'] = 'weighted_layer'
        config['training']['checkpoint_path'] = None

        mock_bert_config_instance = mock_dependencies['mock_bert_config_load'].return_value
        mock_bert_config_instance.output_hidden_states = False 

        train(config)

        assert mock_bert_config_instance.output_hidden_states is True
        mock_dependencies['mock_custom_model'].assert_called_once_with(config=mock_bert_config_instance)
        
        # Verify metrics file creation
        mock_dependencies['mock_generate_artifact_name'].assert_called()

    def test_checkpoint_resume_new_format_with_config(self, mock_dependencies, temp_config_file):
        config = load_config(temp_config_file)
        checkpoint_path = "/fake/checkpoint_epoch1.pth"
        config['training']['resume_from_checkpoint'] = checkpoint_path

        mock_dependencies['mock_os_path_exists'].return_value = True
        
        checkpoint_config = copy.deepcopy(config)
        checkpoint_config['training']['lr'] = 5e-6 
        checkpoint_config['training']['epochs'] = 3 
        
        mock_checkpoint_content = {
            'model_state_dict': {'resumed_state': torch.tensor(2.0)},
            'optimizer_state_dict': {'resumed_opt_state': torch.tensor(2.0)},
            'scheduler_state_dict': {'resumed_sched_state': torch.tensor(2.0)},
            'epoch': 1, 
            'best_f1': 0.75,
            'config': checkpoint_config
        }
        mock_dependencies['mock_torch_load'].return_value = mock_checkpoint_content
        mock_dependencies['mock_re_search'].return_value = MagicMock(group=lambda x: '1' if x in [1,2] else None)

        current_session_output_dir = str(Path(temp_config_file).parent / "new_output_dir_resume" / "models")
        config['model']['output_dir'] = current_session_output_dir

        train(config)

        mock_dependencies['mock_torch_load'].assert_called_once_with(checkpoint_path, map_location=mock_dependencies['mock_model_instance'].to.call_args[0][0])
        mock_dependencies['mock_model_instance'].load_state_dict.assert_called_once_with(mock_checkpoint_content['model_state_dict'])
        mock_dependencies['mock_optimizer_instance'].load_state_dict.assert_called_once_with(mock_checkpoint_content['optimizer_state_dict'])
        mock_dependencies['mock_scheduler_instance'].load_state_dict.assert_called_once_with(mock_checkpoint_content['scheduler_state_dict'])
        
        final_optimizer_call_args = mock_dependencies['mock_adamw'].call_args_list[-1]
        assert final_optimizer_call_args.kwargs['lr'] == checkpoint_config['training']['lr']

        epochs_run = (checkpoint_config['training']['epochs'] - (mock_checkpoint_content['epoch'] + 1) + 1)
        assert mock_dependencies['mock_evaluate'].call_count == epochs_run * 2

        mock_dependencies['mock_torch_save'].assert_called()
        saved_config_in_new_checkpoint = mock_dependencies['mock_torch_save'].call_args[0][0]['config']
        assert saved_config_in_new_checkpoint['model']['output_dir'] == current_session_output_dir
        assert saved_config_in_new_checkpoint['training']['resume_from_checkpoint'] == ""
        
        # Verify metrics file artifact generation was attempted for the correct final epoch and output dir
        mock_dependencies['mock_generate_artifact_name'].assert_any_call(
            base_output_dir=current_session_output_dir,
            model_config_name=ANY, # Be flexible with exact name from potentially loaded config
            loss_function_name=ANY, # Be flexible
            epoch=checkpoint_config['training']['epochs'], # Metrics file uses total target epochs from (potentially loaded) config
            artifact_type='metrics', 
            extension='json',
            timestamp_str=ANY # Timestamp will vary
        )

    def test_checkpoint_resume_old_format(self, mock_dependencies, temp_config_file, tmp_path):
        config = load_config(temp_config_file)
        config['training']['epochs'] = 1 
        # Ensure output_dir is within tmp_path for actual directory creation
        output_dir = tmp_path / "test_output_old_format" / "models"
        config['model']['output_dir'] = str(output_dir)
        checkpoint_path = "/fake/old_checkpoint_epoch0.pth"
        config['training']['resume_from_checkpoint'] = checkpoint_path

        mock_dependencies['mock_os_path_exists'].return_value = True
        mock_old_format_state_dict = {'old_state': torch.tensor(3.0)}
        mock_dependencies['mock_torch_load'].return_value = mock_old_format_state_dict
        mock_epoch_match = MagicMock()
        mock_epoch_match.group = lambda x: '0' if x in [1,2] else None 
        mock_dependencies['mock_re_search'].return_value = mock_epoch_match

        train(config)

        mock_dependencies['mock_model_instance'].load_state_dict.assert_called_once_with(mock_old_format_state_dict)
        mock_dependencies['mock_optimizer_instance'].load_state_dict.assert_not_called()
        mock_dependencies['mock_scheduler_instance'].load_state_dict.assert_not_called()
        
        epochs_run = config['training']['epochs'] - 0 # Resumes from epoch 0+1=1, runs for 'epochs' total
        assert mock_dependencies['mock_evaluate'].call_count == epochs_run * 2
        
        # Verify metrics file artifact generation was attempted
        mock_dependencies['mock_generate_artifact_name'].assert_any_call(
            base_output_dir=config['model']['output_dir'],
            model_config_name=ANY,
            loss_function_name=ANY,
            epoch=config['training']['epochs'],
            artifact_type='metrics', 
            extension='json',
            timestamp_str=ANY
        )

    def test_checkpoint_not_found(self, mock_dependencies, temp_config_file, tmp_path):
        config = load_config(temp_config_file)
        # Ensure output_dir is within tmp_path for actual directory creation
        output_dir = tmp_path / "test_output_not_found" / "models"
        config['model']['output_dir'] = str(output_dir)
        checkpoint_path = "/fake/non_existent.pth"
        config['training']['resume_from_checkpoint'] = checkpoint_path
        mock_dependencies['mock_os_path_exists'].return_value = False

        train(config)
        mock_dependencies['mock_torch_load'].assert_not_called()
        mock_dependencies['mock_model_instance'].load_state_dict.assert_not_called()
        assert mock_dependencies['mock_evaluate'].call_count == config['training']['epochs'] * 2
        
        # Verify metrics file artifact generation was attempted
        mock_dependencies['mock_generate_artifact_name'].assert_any_call(
            base_output_dir=config['model']['output_dir'],
            model_config_name=ANY,
            loss_function_name=ANY,
            epoch=config['training']['epochs'],
            artifact_type='metrics', 
            extension='json',
            timestamp_str=ANY
        )

    def test_checkpoint_load_error(self, mock_dependencies, temp_config_file, tmp_path):
        config = load_config(temp_config_file)
        # Ensure output_dir is within tmp_path for actual directory creation
        output_dir = tmp_path / "test_output_load_error" / "models"
        config['model']['output_dir'] = str(output_dir)
        checkpoint_path = "/fake/corrupt_checkpoint.pth"
        config['training']['resume_from_checkpoint'] = checkpoint_path

        mock_dependencies['mock_os_path_exists'].return_value = True
        mock_dependencies['mock_torch_load'].side_effect = Exception("Corrupt file")

        train(config)
        mock_dependencies['mock_model_instance'].load_state_dict.assert_not_called()
        assert mock_dependencies['mock_evaluate'].call_count == config['training']['epochs'] * 2
        
        # Verify metrics file artifact generation was attempted
        mock_dependencies['mock_generate_artifact_name'].assert_any_call(
            base_output_dir=config['model']['output_dir'],
            model_config_name=ANY,
            loss_function_name=ANY,
            epoch=config['training']['epochs'],
            artifact_type='metrics', 
            extension='json',
            timestamp_str=ANY
        )

    def test_no_improvement_no_save(self, mock_dependencies, temp_config_file):
        config = load_config(temp_config_file)
        config['model']['output_dir'] = str(Path(temp_config_file).parent)
        config['training']['epochs'] = 2 
        config['training']['resume_from_checkpoint'] = None # Corrected from checkpoint_path
        
        mock_evaluate = mock_dependencies['mock_evaluate']
        mock_torch_save = mock_dependencies['mock_torch_save']

        # Ensure enough mock_evaluate results for all evaluations (train + val per epoch)
        # Epoch 1: train_metrics, val_metrics (best_f1 updated, checkpoint saved)
        # Epoch 2: train_metrics, val_metrics (no improvement, no new checkpoint)
        mock_evaluate.side_effect = [
            # Epoch 1 (eval on train, then val)
            {'loss': 0.2, 'accuracy': 0.8, 'f1': 0.75, 'roc_auc': 0.8, 'precision': 0.8, 'recall': 0.8, 'mcc': 0.7},
            {'loss': 0.3, 'accuracy': 0.75, 'f1': 0.7, 'roc_auc': 0.72, 'precision': 0.78, 'recall': 0.72, 'mcc': 0.65},
            # Epoch 2 (eval on train, then val) 
            {'loss': 0.25, 'accuracy': 0.78, 'f1': 0.72, 'roc_auc': 0.78, 'precision': 0.78, 'recall': 0.78, 'mcc': 0.68},
            {'loss': 0.4, 'accuracy': 0.65, 'f1': 0.6, 'roc_auc': 0.62, 'precision': 0.68, 'recall': 0.62, 'mcc': 0.55}
        ]

        train(config)
        # Expect 1 save for the first epoch where f1 improves from 0.0 to 0.7
        # And metrics file save at the end
        assert mock_torch_save.call_count == 1 
        mock_dependencies['mock_generate_artifact_name'].assert_any_call(
            base_output_dir=ANY, model_config_name=ANY, loss_function_name=ANY, 
            epoch=ANY, artifact_type='metrics', extension=ANY, timestamp_str=ANY
        )

    def test_output_dir_creation(self, mock_dependencies, temp_config_file, tmp_path):
        config = load_config(temp_config_file)
        output_dir_str = str(tmp_path / "new_test_output_creation")
        config['model']['output_dir'] = output_dir_str
        
        mock_generate_artifact_name = mock_dependencies['mock_generate_artifact_name']

        train(config)
        
        metrics_file_call_args = None
        for call in mock_generate_artifact_name.call_args_list:
            if call.kwargs.get('artifact_type') == 'metrics':
                metrics_file_call_args = call
                break
        assert metrics_file_call_args is not None, "generate_artifact_name was not called for metrics"

        actual_metrics_path = mock_generate_artifact_name.side_effect(*metrics_file_call_args.args, **metrics_file_call_args.kwargs)

        assert Path(output_dir_str).exists(), f"Base output directory {output_dir_str} was not created."
        assert Path(actual_metrics_path).exists(), f"Metrics file {actual_metrics_path} was not created."

def test_example_mock_access(mock_dependencies, temp_config_file, tmp_path):
    config = load_config(temp_config_file)
    config['model']['output_dir'] = str(tmp_path)
    config['training']['resume_from_checkpoint'] = None

    tokenizer_mock = mock_dependencies['mock_tokenizer']
    download_data_mock = mock_dependencies['mock_download_data']
    create_dataloaders_mock = mock_dependencies['mock_create_dataloaders']
    bert_config_load_mock = mock_dependencies['mock_bert_config_load']
    custom_model_mock = mock_dependencies['mock_custom_model']
    adamw_mock = mock_dependencies['mock_adamw']
    lr_scheduler_mock = mock_dependencies['mock_lr_scheduler']
    evaluate_mock = mock_dependencies['mock_evaluate']
    torch_save_mock = mock_dependencies['mock_torch_save']
    os_path_exists_mock = mock_dependencies['mock_os_path_exists']
    torch_load_mock = mock_dependencies['mock_torch_load']
    re_search_mock = mock_dependencies['mock_re_search']
    # mock_path_mkdir was removed, so remove access to it here

    model_instance_mock = mock_dependencies['mock_model_instance']
    optimizer_instance_mock = mock_dependencies['mock_optimizer_instance']
    scheduler_instance_mock = mock_dependencies['mock_scheduler_instance']
    bert_config_instance_mock = mock_dependencies['mock_bert_config_instance'] 

    os_path_exists_mock.return_value = False 
    evaluate_mock.return_value = {
        'loss': 0.1, 'accuracy': 0.9, 'f1': 0.85, 
        'roc_auc': 0.92, 'precision': 0.88, 'recall': 0.82, 'mcc': 0.75
    }

    train(config)

    tokenizer_mock.assert_called_once()
    model_instance_mock.to.assert_called()
    optimizer_instance_mock.step.assert_called()
    torch_save_mock.assert_called_once()
