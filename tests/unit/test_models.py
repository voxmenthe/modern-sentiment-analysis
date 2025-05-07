import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from transformers import ModernBertConfig 
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutput

from src.models import ModernBertForSentiment
from src.train_utils import SentimentWeightedLoss, SentimentFocalLoss
from src.classifiers import ClassifierHead, ConcatClassifierHead

@pytest.fixture
def base_config():
    config = ModernBertConfig(
        vocab_size=30522,
        hidden_size=32, 
        num_hidden_layers=2, 
        num_attention_heads=2, 
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=1, 
        output_hidden_states=False, 
        classifier_dropout=0.1,
        pad_token_id=0
    )
    config.pooling_strategy = 'cls' 
    config.num_weighted_layers = 2
    config.loss_function = {'name': 'SentimentWeightedLoss', 'params': {}}
    return config

@pytest.fixture
def dummy_input_ids():
    return torch.randint(0, 100, (2, 10)) 

@pytest.fixture
def dummy_attention_mask():
    return torch.ones((2, 10), dtype=torch.long)

@pytest.fixture
def dummy_labels():
    return torch.randint(0, 2, (2,)).float()

@pytest.fixture
def dummy_lengths():
    return torch.tensor([10, 8])

@pytest.fixture
def mock_bert_output_no_hidden_states(base_config):
    batch_size, seq_len, hidden_size = 2, 10, base_config.hidden_size
    last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    pooler_output = torch.randn(batch_size, hidden_size)
    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=last_hidden_state,
        pooler_output=pooler_output,
        hidden_states=None, 
        attentions=None
    )

@pytest.fixture
def mock_bert_output_with_hidden_states(base_config):
    batch_size, seq_len, hidden_size = 2, 10, base_config.hidden_size
    last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    pooler_output = torch.randn(batch_size, hidden_size)
    hidden_states_list = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(base_config.num_hidden_layers + 1)]
    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=last_hidden_state,
        pooler_output=pooler_output,
        hidden_states=tuple(hidden_states_list),
        attentions=None
    )

class TestModernBertForSentiment:

    def test_initialization_default(self, base_config):
        model = ModernBertForSentiment(base_config)
        assert isinstance(model.bert, nn.Module) 
        assert model.pooling_strategy == 'cls'
        assert isinstance(model.classifier, ClassifierHead)
        assert model.classifier.dense1.in_features == base_config.hidden_size 
        assert model.classifier.out_proj.out_features == base_config.num_labels
        assert model.config.output_hidden_states is False 

    @pytest.mark.parametrize("pooling_strategy, expected_classifier_type, needs_hidden_states", [
        ('cls', ClassifierHead, False),
        ('mean', ClassifierHead, False),
        ('cls_mean_concat', ConcatClassifierHead, False),
        ('weighted_layer', ClassifierHead, True),
        ('cls_weighted_concat', ConcatClassifierHead, True)
    ])
    def test_initialization_pooling_strategies(self, base_config, pooling_strategy, expected_classifier_type, needs_hidden_states):
        base_config.pooling_strategy = pooling_strategy
        base_config.output_hidden_states = needs_hidden_states 

        model = ModernBertForSentiment(base_config)

        assert isinstance(model.classifier, expected_classifier_type)
        assert model.config.output_hidden_states == needs_hidden_states

        if pooling_strategy in ['cls_mean_concat', 'cls_weighted_concat']:
            assert model.classifier.initial_projection.in_features == base_config.hidden_size * 2
            assert model.classifier.dense1.in_features == base_config.hidden_size 
        else:
            assert model.classifier.dense1.in_features == base_config.hidden_size 

        if pooling_strategy in ['weighted_layer', 'cls_weighted_concat']:
            assert hasattr(model, 'layer_weights')
            assert model.layer_weights.shape == (base_config.num_weighted_layers,)

    def test_initialization_weighted_pooling_error(self, base_config):
        base_config.pooling_strategy = 'weighted_layer'
        base_config.output_hidden_states = False 
        with pytest.raises(ValueError, match="output_hidden_states must be True in BertConfig for weighted_layer pooling"):
            ModernBertForSentiment(base_config)

    def test_initialization_custom_loss(self, base_config):
        base_config.loss_function = {'name': 'SentimentFocalLoss', 'params': {'gamma_focal': 1.5, 'label_smoothing_epsilon': 0.05}}
        model = ModernBertForSentiment(base_config)
        assert isinstance(model.loss_fct, SentimentFocalLoss)
        assert model.loss_fct.gamma_focal == 1.5
        assert model.loss_fct.label_smoothing_epsilon == 0.05

    def test_initialization_unsupported_loss(self, base_config):
        base_config.loss_function = {'name': 'UnsupportedLoss', 'params': {}}
        with pytest.raises(ValueError, match="Unsupported loss function: UnsupportedLoss"):
            ModernBertForSentiment(base_config)

    def test_forward_pass_cls_pooling(self, base_config, dummy_input_ids):
        base_config.pooling_strategy = 'cls'
        model = ModernBertForSentiment(base_config)
        model.classifier.forward = Mock(return_value=torch.randn(dummy_input_ids.shape[0], model.num_labels)) 

        mock_bert_output = BaseModelOutput(
            last_hidden_state=torch.randn(dummy_input_ids.shape[0], dummy_input_ids.shape[1], base_config.hidden_size)
        )
        model.bert.forward = Mock(return_value=mock_bert_output)

        outputs = model(input_ids=dummy_input_ids)
        assert outputs.logits is not None
        model.classifier.forward.assert_called_once() 
        args, _ = model.classifier.forward.call_args
        assert args[0].shape == (dummy_input_ids.shape[0], base_config.hidden_size)

    def test_forward_pass_mean_pooling(self, base_config, dummy_input_ids, dummy_attention_mask):
        base_config.pooling_strategy = 'mean'
        model = ModernBertForSentiment(base_config)
        model.classifier.forward = Mock(return_value=torch.randn(dummy_input_ids.shape[0], model.num_labels)) 

        mock_bert_output = BaseModelOutput(
            last_hidden_state=torch.randn(dummy_input_ids.shape[0], dummy_input_ids.shape[1], base_config.hidden_size)
        )
        model.bert.forward = Mock(return_value=mock_bert_output)

        outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
        assert outputs.logits is not None
        model.classifier.forward.assert_called_once()
        args, _ = model.classifier.forward.call_args
        assert args[0].shape == (dummy_input_ids.shape[0], base_config.hidden_size)

    def test_forward_pass_weighted_layer_pooling(self, base_config, dummy_input_ids):
        base_config.pooling_strategy = 'weighted_layer'
        base_config.output_hidden_states = True 
        model = ModernBertForSentiment(base_config)
        model.classifier.forward = Mock(return_value=torch.randn(dummy_input_ids.shape[0], model.num_labels)) 

        mock_all_hidden_states = [torch.randn(*dummy_input_ids.shape[:2] + (base_config.hidden_size,)) for _ in range(base_config.num_hidden_layers + 1)]

        mock_bert_output = BaseModelOutput(
            last_hidden_state=mock_all_hidden_states[-1], 
            hidden_states=tuple(mock_all_hidden_states) 
        )
        model.bert.forward = Mock(return_value=mock_bert_output)

        outputs = model(input_ids=dummy_input_ids)
        assert outputs.logits is not None
        model.classifier.forward.assert_called_once()
        args, _ = model.classifier.forward.call_args
        assert args[0].shape == (dummy_input_ids.shape[0], base_config.hidden_size)

    def test_forward_pass_weighted_pooling_value_error_if_no_hidden_states_output(self, base_config, dummy_input_ids):
        base_config.pooling_strategy = 'weighted_layer'
        base_config.output_hidden_states = True 
        model = ModernBertForSentiment(base_config)

        mock_bert_output_no_hs = BaseModelOutput(
            last_hidden_state=torch.randn(dummy_input_ids.shape[0], dummy_input_ids.shape[1], base_config.hidden_size),
            hidden_states=None 
        )
        model.bert.forward = Mock(return_value=mock_bert_output_no_hs)

        with pytest.raises(ValueError, match="Weighted layer pooling requires output_hidden_states=True and hidden_states in BERT output."):
            model(input_ids=dummy_input_ids)

    def test_forward_pass_with_labels(self, base_config, dummy_input_ids, dummy_labels, dummy_lengths):
        model = ModernBertForSentiment(base_config)
        model.loss_fct.forward = Mock(return_value=torch.tensor(0.5, requires_grad=True)) 

        mock_bert_output = BaseModelOutput(
            last_hidden_state=torch.randn(dummy_input_ids.shape[0], dummy_input_ids.shape[1], base_config.hidden_size)
        )
        model.bert.forward = Mock(return_value=mock_bert_output)

        outputs = model(input_ids=dummy_input_ids, labels=dummy_labels, lengths=dummy_lengths)
        assert outputs.loss is not None
        model.loss_fct.forward.assert_called_once()
        args, _ = model.loss_fct.forward.call_args
        assert args[0].shape == (dummy_input_ids.shape[0],) 
        assert args[1] is dummy_labels
        assert args[2] is dummy_lengths

    def test_forward_pass_labels_no_lengths_error(self, base_config, dummy_input_ids, dummy_labels):
        model = ModernBertForSentiment(base_config)
        mock_bert_output = BaseModelOutput(
            last_hidden_state=torch.randn(dummy_input_ids.shape[0], dummy_input_ids.shape[1], base_config.hidden_size)
        )
        model.bert.forward = Mock(return_value=mock_bert_output)

        with pytest.raises(ValueError, match="lengths must be provided when labels are specified for loss calculation."):
            model(input_ids=dummy_input_ids, labels=dummy_labels) 

    def test_weighted_layer_pool_method(self, base_config):
        base_config.pooling_strategy = 'weighted_layer' 
        base_config.output_hidden_states = True     
        model = ModernBertForSentiment(base_config)

        batch_size, seq_len, hidden_size = 2, 10, base_config.hidden_size
        all_hidden_states = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(base_config.num_hidden_layers + 1)]
        all_hidden_states_stack = torch.stack(all_hidden_states[-base_config.num_weighted_layers:], dim=0)

        assert hasattr(model, 'layer_weights'), "Model should have 'layer_weights' attribute for weighted_layer pooling"
        assert model.layer_weights.shape == (base_config.num_weighted_layers,)

        with patch('torch.nn.functional.softmax') as mock_softmax:
            mock_softmax.return_value = torch.ones_like(model.layer_weights) / model.num_weighted_layers

            pooled_output = model._weighted_layer_pool(all_hidden_states) 

            mock_softmax.assert_called_once()
            assert torch.equal(mock_softmax.call_args[0][0], model.layer_weights)

            assert pooled_output.shape == (batch_size, hidden_size)

    def test_unsupported_pooling_strategy(self, base_config, dummy_input_ids):
        base_config.pooling_strategy = "non_existent_pooling"
        model = ModernBertForSentiment(base_config)
        model.bert.forward = Mock(return_value=BaseModelOutput(last_hidden_state=torch.randn(dummy_input_ids.shape[0], dummy_input_ids.shape[1], base_config.hidden_size)))
        with pytest.raises(ValueError, match="Unknown pooling_strategy: non_existent_pooling"):
            model(input_ids=dummy_input_ids)
