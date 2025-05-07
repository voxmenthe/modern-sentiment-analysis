from transformers import ModernBertModel, ModernBertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch
from src.train_utils import SentimentWeightedLoss, SentimentFocalLoss
import torch.nn.functional as F

from src.classifiers import ClassifierHead, ConcatClassifierHead


class ModernBertForSentiment(ModernBertPreTrainedModel):
    """ModernBERT encoder with a dynamically configurable classification head and pooling strategy."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = ModernBertModel(config) # Base BERT model, config may have output_hidden_states=True

        # Store pooling strategy from config
        self.pooling_strategy = getattr(config, 'pooling_strategy', 'cls') # Default to 'cls'
        self.num_weighted_layers = getattr(config, 'num_weighted_layers', 4)

        if self.pooling_strategy in ['weighted_layer', 'cls_weighted_concat'] and not config.output_hidden_states:
            # This check is more of an assertion; train.py should set output_hidden_states=True
            raise ValueError(
                "output_hidden_states must be True in BertConfig for weighted_layer pooling."
            )

        # Initialize weights for weighted layer pooling
        if self.pooling_strategy in ['weighted_layer', 'cls_weighted_concat']:
            # num_weighted_layers specifies how many *top* layers of BERT to use.
            # If num_weighted_layers is e.g. 4, we use the last 4 layers.
            self.layer_weights = nn.Parameter(torch.ones(self.num_weighted_layers) / self.num_weighted_layers)

        # Determine classifier input size and choose head
        classifier_input_size = config.hidden_size
        if self.pooling_strategy in ['cls_mean_concat', 'cls_weighted_concat']:
            classifier_input_size = config.hidden_size * 2
        
        # Dropout for features fed into the classifier head
        classifier_dropout_prob = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.features_dropout = nn.Dropout(classifier_dropout_prob)

        # Select the appropriate classifier head based on input feature dimension
        if classifier_input_size == config.hidden_size:
            self.classifier = ClassifierHead(
                hidden_size=config.hidden_size, # input_size for ClassifierHead is just hidden_size
                num_labels=config.num_labels,
                dropout_prob=classifier_dropout_prob
            )
        elif classifier_input_size == config.hidden_size * 2:
            self.classifier = ConcatClassifierHead(
                input_size=config.hidden_size * 2,
                hidden_size=config.hidden_size, # Internal hidden size of the head
                num_labels=config.num_labels,
                dropout_prob=classifier_dropout_prob
            )
        else:
            # This case should ideally not be reached with current strategies
            raise ValueError(f"Unexpected classifier_input_size: {classifier_input_size}")

        # Initialize loss function based on config
        loss_config = getattr(config, 'loss_function', {'name': 'SentimentWeightedLoss', 'params': {}})
        loss_name = loss_config.get('name', 'SentimentWeightedLoss')
        loss_params = loss_config.get('params', {})

        if loss_name == "SentimentWeightedLoss":
            self.loss_fct = SentimentWeightedLoss() # SentimentWeightedLoss takes no arguments
        elif loss_name == "SentimentFocalLoss":
            # Ensure only relevant params are passed, or that loss_params is structured correctly for SentimentFocalLoss
            # For SentimentFocalLoss, expected params are 'gamma_focal' and 'label_smoothing_epsilon'
            self.loss_fct = SentimentFocalLoss(**loss_params)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

        self.post_init() # Initialize weights and apply final processing

    def _mean_pool(self, last_hidden_state, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(last_hidden_state[:, :, 0]) # Assuming first dim of last hidden state is token ids
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _weighted_layer_pool(self, all_hidden_states):
        # all_hidden_states includes embeddings + output of each layer.
        # We want the outputs of the last num_weighted_layers.
        # Example: 12 layers -> all_hidden_states have 13 items (embeddings + 12 layers)
        # num_weighted_layers = 4 -> use layers 9, 10, 11, 12 (indices -4, -3, -2, -1)
        layers_to_weigh = torch.stack(all_hidden_states[-self.num_weighted_layers:], dim=0)
        # layers_to_weigh shape: (num_weighted_layers, batch_size, sequence_length, hidden_size)
        
        # Normalize weights to sum to 1 (softmax or simple division)
        normalized_weights = F.softmax(self.layer_weights, dim=-1)
        
        # Weighted sum across layers
        # Reshape weights for broadcasting: (num_weighted_layers, 1, 1, 1)
        weighted_hidden_states = layers_to_weigh * normalized_weights.view(-1, 1, 1, 1)
        weighted_sum_hidden_states = torch.sum(weighted_hidden_states, dim=0)
        # weighted_sum_hidden_states shape: (batch_size, sequence_length, hidden_size)
        
        # Pool the result (e.g., take [CLS] token of this weighted sum)
        return weighted_sum_hidden_states[:, 0] # Return CLS token of the weighted sum

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        lengths=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            output_hidden_states=self.config.output_hidden_states # Controlled by train.py
        )

        last_hidden_state = bert_outputs[0] # Or bert_outputs.last_hidden_state
        pooled_features = None

        if self.pooling_strategy == 'cls':
            pooled_features = last_hidden_state[:, 0] # CLS token
        elif self.pooling_strategy == 'mean':
            pooled_features = self._mean_pool(last_hidden_state, attention_mask)
        elif self.pooling_strategy == 'cls_mean_concat':
            cls_output = last_hidden_state[:, 0]
            mean_output = self._mean_pool(last_hidden_state, attention_mask)
            pooled_features = torch.cat((cls_output, mean_output), dim=1)
        elif self.pooling_strategy == 'weighted_layer':
            if not self.config.output_hidden_states or bert_outputs.hidden_states is None:
                raise ValueError("Weighted layer pooling requires output_hidden_states=True and hidden_states in BERT output.")
            all_hidden_states = bert_outputs.hidden_states
            pooled_features = self._weighted_layer_pool(all_hidden_states)
        elif self.pooling_strategy == 'cls_weighted_concat':
            if not self.config.output_hidden_states or bert_outputs.hidden_states is None:
                raise ValueError("Weighted layer pooling requires output_hidden_states=True and hidden_states in BERT output.")
            cls_output = last_hidden_state[:, 0]
            all_hidden_states = bert_outputs.hidden_states
            weighted_output = self._weighted_layer_pool(all_hidden_states)
            pooled_features = torch.cat((cls_output, weighted_output), dim=1)
        else:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")

        pooled_features = self.features_dropout(pooled_features)
        logits = self.classifier(pooled_features)

        loss = None
        if labels is not None:
            if lengths is None:
                raise ValueError("lengths must be provided when labels are specified for loss calculation.")
            loss = self.loss_fct(logits.squeeze(-1), labels, lengths)

        if not return_dict:
            # Ensure 'outputs' from BERT is appropriately handled. If it's a tuple:            
            bert_model_outputs = bert_outputs[1:] if isinstance(bert_outputs, tuple) else (bert_outputs.hidden_states, bert_outputs.attentions)
            output = (logits,) + bert_model_outputs
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )