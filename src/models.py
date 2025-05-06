from transformers import ModernBertModel, ModernBertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch
from train_utils import SentimentWeightedLoss
import torch.nn.functional as F


class EnhancedClassifierHead(nn.Module):
    """A 3-layer classifier head with GELU, LayerNorm, and Skip Connections."""
    def __init__(self, hidden_size, num_labels, dropout_prob):
        super().__init__()
        # Layer 1
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)

        # Layer 2
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)

        # Output Layer
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        # Layer 1 + Skip
        identity1 = features
        x = self.norm1(features)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = x + identity1 # Skip connection 1

        # Layer 2 + Skip
        identity2 = x
        x = self.norm2(x)
        x = self.dense2(x)
        x = self.activation(x) # Re-use activation
        x = self.dropout2(x)
        x = x + identity2 # Skip connection 2

        # Output Layer
        logits = self.out_proj(x)
        return logits


class ModernBertForSentiment(ModernBertPreTrainedModel):
    """ModernBERT encoder with a custom ENHANCED classification head (single logit)
       and custom loss calculation.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels # Store num_labels from config
        self.bert = ModernBertModel(config)  # Instantiate the base BERT model

        # Define custom head components
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # This dropout applies to the BERT output BEFORE the head
        self.bert_output_dropout = nn.Dropout(classifier_dropout)
        # Use the new enhanced classifier head
        self.classifier = EnhancedClassifierHead(
            config.hidden_size,
            config.num_labels,
            classifier_dropout # Pass dropout to the head as well
        )

        # IMPORTANT: Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        lengths=None, # Keep lengths for custom loss
        return_dict=None,
        **kwargs # Accept other potential args from trainer
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs
        )

        # Use the hidden state of the [CLS] token (index 0)
        cls_hidden_state = outputs[0][:, 0]
        # Apply dropout to BERT output first
        pooled_output = self.bert_output_dropout(cls_hidden_state)
        # Pass through the enhanced classifier head
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Use the custom weighted loss
            if lengths is None:
                raise ValueError("lengths must be provided when labels are specified for SentimentWeightedLoss")
            loss_fct = SentimentWeightedLoss()
            # Ensure labels are float if needed by loss (depends on loss impl.)
            # Assuming loss_fct handles type conversion or expects raw logits/int labels
            loss = loss_fct(logits.squeeze(-1), labels, lengths)

        if not return_dict:
            output = (logits,) + outputs[1:] # Include hidden states/attentions if returned by base
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )