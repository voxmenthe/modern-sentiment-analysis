import pytest
import torch
import torch.nn as nn

from src.classifiers import ClassifierHead, ConcatClassifierHead

@pytest.fixture
def hidden_size():
    return 64 # Example hidden size

@pytest.fixture
def dropout_prob():
    return 0.1

@pytest.fixture
def batch_size():
    return 4

class TestClassifierHead:
    def test_initialization(self, hidden_size, dropout_prob): 
        fixed_num_labels = 1 
        ch = ClassifierHead(hidden_size, fixed_num_labels, dropout_prob)
        assert isinstance(ch.dense1, nn.Linear) 
        assert ch.dense1.in_features == hidden_size
        assert ch.dense1.out_features == hidden_size 
        assert isinstance(ch.dropout1, nn.Dropout) 
        assert ch.dropout1.p == dropout_prob
        assert ch.out_proj.in_features == hidden_size 
        assert ch.out_proj.out_features == fixed_num_labels

    def test_forward_pass(self, hidden_size, dropout_prob, batch_size): 
        fixed_num_labels = 1
        ch = ClassifierHead(hidden_size, fixed_num_labels, dropout_prob)
        dummy_input = torch.randn(batch_size, hidden_size)
        output = ch(dummy_input)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, fixed_num_labels)

    def test_forward_pass_eval_mode(self, hidden_size, dropout_prob, batch_size): 
        fixed_num_labels = 1
        ch = ClassifierHead(hidden_size, fixed_num_labels, dropout_prob)
        ch.eval() 
        dummy_input = torch.randn(batch_size, hidden_size)
        output = ch(dummy_input)
        output2 = ch(dummy_input)
        assert torch.allclose(output, output2)
        assert output.shape == (batch_size, fixed_num_labels)

class TestConcatClassifierHead:
    def test_initialization(self, hidden_size, dropout_prob): 
        fixed_num_labels = 1 
        input_concat_size = hidden_size * 2
        cch = ConcatClassifierHead(input_concat_size, hidden_size, fixed_num_labels, dropout_prob)
        
        assert isinstance(cch.initial_projection, nn.Linear)
        assert cch.initial_projection.in_features == input_concat_size
        assert cch.initial_projection.out_features == hidden_size
        assert isinstance(cch.initial_dropout, nn.Dropout)
        assert cch.initial_dropout.p == dropout_prob

        assert isinstance(cch.dense1, nn.Linear)
        assert cch.dense1.in_features == hidden_size
        assert cch.dense1.out_features == hidden_size 
        assert isinstance(cch.dropout1, nn.Dropout) 
        assert cch.dropout1.p == dropout_prob
        
        assert isinstance(cch.dense2, nn.Linear)
        assert cch.dense2.in_features == hidden_size 
        assert cch.dense2.out_features == hidden_size 
        assert isinstance(cch.dropout2, nn.Dropout) 
        assert cch.dropout2.p == dropout_prob

        assert cch.out_proj.in_features == hidden_size 
        assert cch.out_proj.out_features == fixed_num_labels


    def test_forward_pass(self, hidden_size, dropout_prob, batch_size): 
        fixed_num_labels = 1
        input_concat_size = hidden_size * 2
        cch = ConcatClassifierHead(input_concat_size, hidden_size, fixed_num_labels, dropout_prob)
        dummy_input = torch.randn(batch_size, input_concat_size)
        output = cch(dummy_input)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, fixed_num_labels)

    def test_forward_pass_eval_mode(self, hidden_size, dropout_prob, batch_size): 
        fixed_num_labels = 1
        input_concat_size = hidden_size * 2
        cch = ConcatClassifierHead(input_concat_size, hidden_size, fixed_num_labels, dropout_prob)
        cch.eval() 
        dummy_input = torch.randn(batch_size, input_concat_size)
        output = cch(dummy_input)
        output2 = cch(dummy_input)
        assert torch.allclose(output, output2)
        assert output.shape == (batch_size, fixed_num_labels)
