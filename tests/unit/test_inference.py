import pytest
import torch
from unittest.mock import patch, MagicMock, mock_open
import yaml
import os

# Class to be tested
from src.inference import SentimentInference
# Depending on where ModernBertConfig and ModernBertForSentiment are, they might need to be imported
# for spec_set in MagicMock, or we can just mock them as generic MagicMock objects.
# from src.models import ModernBertForSentiment # If needed for spec
# from transformers import ModernBertConfig # If needed for spec

@pytest.fixture
def default_config_data():
    '''Provides a default config dictionary for tests.'''
    return {
        'model': {
            'name': 'answerdotai/ModernBERT-base',
            'output_dir': 'checkpoints_test',
            'max_length': 128,
            'dropout': 0.15,
            'pooling_strategy': 'mean',
            'num_weighted_layers': 2,
            'loss_function': {'name': 'BCELoss', 'params': {}},
            # num_labels is set to 1 inside SentimentInference init for bert_config
        },
        'inference': {
            'model_path': 'checkpoints_test/best_model.pt',
            'max_length': 120, # Test inference specific max_length
        }
    }

@pytest.fixture
def mock_tokenizer_instance():
    '''Returns a mock tokenizer instance.'''
    tokenizer = MagicMock()
    # Mock the __call__ method to simulate tokenization
    tokenizer.return_value = {
        'input_ids': torch.randint(0, 100, (1, 5)), # batch_size 1, seq_len 5
        'attention_mask': torch.ones(1, 5)
    }
    return tokenizer

@pytest.fixture
def mock_bert_config_instance():
    '''Returns a mock BertConfig instance.'''
    config = MagicMock()
    config.classifier_dropout = 0.1 # Default value, will be overridden
    config.pooling_strategy = 'cls' # Default
    config.num_weighted_layers = 1 # Default
    config.loss_function = {'name': 'SomeLoss'}
    config.num_labels = 2 # Default, should be set to 1
    return config

@pytest.fixture
def mock_sentiment_model_instance():
    '''Returns a mock Sentiment Model instance.'''
    model = MagicMock()
    model.eval = MagicMock()
    model.load_state_dict = MagicMock()
    # Mock the __call__ method to simulate model inference
    # Default to positive sentiment
    model.return_value = {'logits': torch.tensor([[0.6]])} # (batch_size, num_labels=1 for sigmoid)
    return model

# --- Test Class for SentimentInference ---

class TestSentimentInference:

    @patch('src.inference.yaml.safe_load')
    @patch('src.inference.open', new_callable=mock_open)
    @patch('src.inference.AutoTokenizer.from_pretrained')
    @patch('src.inference.ModernBertConfig.from_pretrained')
    @patch('src.inference.ModernBertForSentiment') # Mock the class constructor
    @patch('src.inference.torch.load')
    @patch('src.inference.os.path.join', side_effect=lambda *args: "/".join(args)) # Simple mock for os.path.join
    def test_initialization_defaults_and_direct_state_dict(
        self, mock_os_join, mock_torch_load, mock_ModernBertForSentiment, 
        mock_ModernBertConfig_from_pretrained, mock_AutoTokenizer_from_pretrained, 
        mock_file_open, mock_yaml_safe_load, 
        default_config_data, mock_tokenizer_instance, mock_bert_config_instance, mock_sentiment_model_instance
    ):
        '''Test basic initialization with default config and direct state_dict in checkpoint.'''
        mock_yaml_safe_load.return_value = default_config_data
        mock_AutoTokenizer_from_pretrained.return_value = mock_tokenizer_instance
        mock_ModernBertConfig_from_pretrained.return_value = mock_bert_config_instance
        mock_ModernBertForSentiment.return_value = mock_sentiment_model_instance
        
        # Case 1: torch.load returns the state_dict directly
        mock_model_state_dict = {'param1': torch.tensor([1.0])}
        mock_torch_load.return_value = mock_model_state_dict

        inference_instance = SentimentInference(config_path="dummy_config.yaml")

        mock_file_open.assert_called_once_with("dummy_config.yaml", 'r')
        mock_yaml_safe_load.assert_called_once()
        mock_AutoTokenizer_from_pretrained.assert_called_once_with(default_config_data['model']['name'])
        mock_ModernBertConfig_from_pretrained.assert_called_once_with(default_config_data['model']['name'])
        
        # Check config overrides on mock_bert_config_instance
        assert mock_bert_config_instance.classifier_dropout == default_config_data['model']['dropout']
        assert mock_bert_config_instance.pooling_strategy == default_config_data['model']['pooling_strategy']
        assert mock_bert_config_instance.num_weighted_layers == default_config_data['model']['num_weighted_layers']
        assert mock_bert_config_instance.loss_function == default_config_data['model']['loss_function']
        assert mock_bert_config_instance.num_labels == 1 # Hardcoded in SentimentInference

        mock_ModernBertForSentiment.assert_called_once_with(mock_bert_config_instance)
        mock_torch_load.assert_called_once_with(default_config_data['inference']['model_path'], map_location=torch.device('cpu'))
        mock_sentiment_model_instance.load_state_dict.assert_called_once_with(mock_model_state_dict)
        mock_sentiment_model_instance.eval.assert_called_once()
        assert inference_instance.max_length == default_config_data['inference']['max_length']

    @patch('src.inference.yaml.safe_load')
    @patch('src.inference.open', new_callable=mock_open)
    @patch('src.inference.AutoTokenizer.from_pretrained')
    @patch('src.inference.ModernBertConfig.from_pretrained')
    @patch('src.inference.ModernBertForSentiment')
    @patch('src.inference.torch.load')
    def test_initialization_with_model_state_dict_in_checkpoint(
        self, mock_torch_load, mock_ModernBertForSentiment, 
        mock_ModernBertConfig_from_pretrained, mock_AutoTokenizer_from_pretrained, 
        mock_file_open, mock_yaml_safe_load,
        default_config_data, mock_tokenizer_instance, mock_bert_config_instance, mock_sentiment_model_instance
    ):
        '''Test initialization when checkpoint contains 'model_state_dict'.'''
        mock_yaml_safe_load.return_value = default_config_data
        mock_AutoTokenizer_from_pretrained.return_value = mock_tokenizer_instance
        mock_ModernBertConfig_from_pretrained.return_value = mock_bert_config_instance
        mock_ModernBertForSentiment.return_value = mock_sentiment_model_instance

        # Case 2: torch.load returns a dict containing 'model_state_dict'
        mock_model_state_dict = {'param1': torch.tensor([1.0])}
        checkpoint_dict = {'model_state_dict': mock_model_state_dict, 'optimizer_state_dict': {}, 'epoch': 1}
        mock_torch_load.return_value = checkpoint_dict

        SentimentInference(config_path="dummy_config.yaml") # Instance creation
        mock_sentiment_model_instance.load_state_dict.assert_called_once_with(mock_model_state_dict)

    @patch('src.inference.yaml.safe_load')
    @patch('src.inference.open', new_callable=mock_open)
    @patch('src.inference.AutoTokenizer.from_pretrained')
    @patch('src.inference.ModernBertConfig.from_pretrained')
    @patch('src.inference.ModernBertForSentiment')
    @patch('src.inference.torch.load')
    def test_initialization_fallbacks_for_missing_config(
        self, mock_torch_load, mock_ModernBertForSentiment, 
        mock_ModernBertConfig_from_pretrained, mock_AutoTokenizer_from_pretrained, 
        mock_file_open, mock_yaml_safe_load, 
        mock_tokenizer_instance, mock_bert_config_instance, mock_sentiment_model_instance
    ):
        '''Test initialization fallbacks when some config keys are missing.'''
        # Create a config with some missing keys
        minimal_config = {
            'model': {'name': 'some-base-model'}, # output_dir, max_length, dropout etc. missing
            'inference': {} # model_path, max_length missing
        }
        mock_yaml_safe_load.return_value = minimal_config
        mock_AutoTokenizer_from_pretrained.return_value = mock_tokenizer_instance
        mock_ModernBertConfig_from_pretrained.return_value = mock_bert_config_instance
        mock_ModernBertForSentiment.return_value = mock_sentiment_model_instance
        mock_torch_load.return_value = {'weights': 1}

        # Expected fallbacks (some are hardcoded in SentimentInference, some from base mock_bert_config_instance)
        expected_model_path = os.path.join('checkpoints', 'best_model.pt') # default output_dir is 'checkpoints'
        expected_max_length = 256 # default from SentimentInference

        inference_instance = SentimentInference()

        mock_ModernBertConfig_from_pretrained.assert_called_once_with('some-base-model')
        # Check that bert_config attributes were attempted to be set (using defaults from base or model)
        assert mock_bert_config_instance.classifier_dropout == 0.1 # From original mock_bert_config default
        assert mock_bert_config_instance.pooling_strategy == 'cls' # Default in SentimentInference
        assert mock_bert_config_instance.num_weighted_layers == 4 # Default in SentimentInference
        assert mock_bert_config_instance.loss_function == {'name': 'SentimentWeightedLoss', 'params': {}} # Default in SentimentInference
        assert mock_bert_config_instance.num_labels == 1
        
        mock_torch_load.assert_called_with(expected_model_path, map_location=torch.device('cpu'))
        assert inference_instance.max_length == expected_max_length

    # --- Predict method tests ---
    @patch('src.inference.yaml.safe_load')
    @patch('src.inference.open', new_callable=mock_open)
    @patch('src.inference.AutoTokenizer.from_pretrained')
    @patch('src.inference.ModernBertConfig.from_pretrained')
    @patch('src.inference.ModernBertForSentiment')
    @patch('src.inference.torch.load')
    def test_predict_positive_sentiment(
        self, mock_torch_load, mock_ModernBertForSentiment, 
        mock_ModernBertConfig_from_pretrained, mock_AutoTokenizer_from_pretrained, 
        mock_file_open, mock_yaml_safe_load, 
        default_config_data, mock_tokenizer_instance, mock_bert_config_instance, mock_sentiment_model_instance
    ):
        '''Test predict method for positive sentiment.'''
        mock_yaml_safe_load.return_value = default_config_data
        mock_AutoTokenizer_from_pretrained.return_value = mock_tokenizer_instance
        mock_ModernBertConfig_from_pretrained.return_value = mock_bert_config_instance
        mock_ModernBertForSentiment.return_value = mock_sentiment_model_instance
        mock_torch_load.return_value = {'weights': 1}

        # Configure model to output positive sentiment logits
        positive_logits = torch.tensor([[0.8]]) # sigmoid(0.8) > 0.5
        mock_sentiment_model_instance.return_value = {'logits': positive_logits}

        inference_instance = SentimentInference()
        result = inference_instance.predict("This is a great movie!")

        expected_prob = torch.sigmoid(positive_logits).item()
        assert result['sentiment'] == "positive"
        assert result['confidence'] == pytest.approx(expected_prob)
        mock_tokenizer_instance.assert_called_once_with(
            "This is a great movie!", 
            return_tensors="pt", 
            truncation=True, 
            max_length=default_config_data['inference']['max_length']
        )
        mock_sentiment_model_instance.assert_called_once_with(
            input_ids=mock_tokenizer_instance.return_value['input_ids'],
            attention_mask=mock_tokenizer_instance.return_value['attention_mask']
        )

    @patch('src.inference.yaml.safe_load')
    @patch('src.inference.open', new_callable=mock_open)
    @patch('src.inference.AutoTokenizer.from_pretrained')
    @patch('src.inference.ModernBertConfig.from_pretrained')
    @patch('src.inference.ModernBertForSentiment')
    @patch('src.inference.torch.load')
    def test_predict_negative_sentiment(
        self, mock_torch_load, mock_ModernBertForSentiment, 
        mock_ModernBertConfig_from_pretrained, mock_AutoTokenizer_from_pretrained, 
        mock_file_open, mock_yaml_safe_load, 
        default_config_data, mock_tokenizer_instance, mock_bert_config_instance, mock_sentiment_model_instance
    ):
        '''Test predict method for negative sentiment.'''
        mock_yaml_safe_load.return_value = default_config_data
        mock_AutoTokenizer_from_pretrained.return_value = mock_tokenizer_instance
        mock_ModernBertConfig_from_pretrained.return_value = mock_bert_config_instance
        mock_ModernBertForSentiment.return_value = mock_sentiment_model_instance
        mock_torch_load.return_value = {'weights': 1}

        # Configure model to output negative sentiment logits
        negative_logits = torch.tensor([[-0.5]]) # sigmoid(-0.5) < 0.5
        mock_sentiment_model_instance.return_value = {'logits': negative_logits}

        inference_instance = SentimentInference()
        result = inference_instance.predict("This is a terrible movie!")

        expected_prob = torch.sigmoid(negative_logits).item()
        assert result['sentiment'] == "negative"
        assert result['confidence'] == pytest.approx(expected_prob)
        mock_tokenizer_instance.assert_called_once_with(
            "This is a terrible movie!", 
            return_tensors="pt", 
            truncation=True, 
            max_length=default_config_data['inference']['max_length']
        )
