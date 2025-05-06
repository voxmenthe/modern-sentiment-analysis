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


class AdvancedClassifierHead(nn.Module):
    """
    A classifier head using FFN-style expansion (input -> 4*hidden -> hidden -> labels).
    Takes concatenated CLS + Mean Pooled features as input.
    """
    def __init__(self, input_size, hidden_size, num_labels, dropout_prob):
        super().__init__()
        intermediate_size = hidden_size * 4 # FFN expansion factor

        # Layer 1 (Expansion)
        self.norm1 = nn.LayerNorm(input_size)
        self.dense1 = nn.Linear(input_size, intermediate_size)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)

        # Layer 2 (Projection back down)
        self.norm2 = nn.LayerNorm(intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        # Activation and Dropout are applied after projection
        self.dropout2 = nn.Dropout(dropout_prob)

        # Output Layer
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        # Layer 1
        x = self.norm1(features)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.norm2(x)
        x = self.dense2(x)
        x = self.activation(x) # Activation after projection
        x = self.dropout2(x)

        # Output Layer
        logits = self.out_proj(x)
        return logits


class ModernBertForSentiment(ModernBertPreTrainedModel):
    """ModernBERT encoder with a custom ADVANCED classification head (single logit)
       using CLS+Mean pooling and FFN expansion, plus custom loss.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels # Store num_labels from config
        self.bert = ModernBertModel(config)  # Instantiate the base BERT model

        # Define custom head components
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # This dropout applies to the BERT output(s) BEFORE the head
        self.bert_output_dropout = nn.Dropout(classifier_dropout)
        
        # Use the new ADVANCED classifier head
        # Input size is doubled because we concatenate CLS and Mean Pool outputs
        self.classifier = AdvancedClassifierHead(
            input_size=config.hidden_size * 2, 
            hidden_size=config.hidden_size, # Internal hidden size of the head
            num_labels=config.num_labels,
            dropout_prob=classifier_dropout
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
            # We need the last hidden state for mean pooling
            output_hidden_states=False, # No need for all hidden states
            **kwargs
        )

        last_hidden_state = outputs[0] # shape: (batch_size, sequence_length, hidden_size)

        # --- Calculate Pooling --- 
        # 1. CLS Token Output
        cls_hidden_state = last_hidden_state[:, 0]
        
        # 2. Mean Pooling of non-padding tokens
        if attention_mask is None:
             # If no mask, assume all tokens are valid (shouldn't happen with tokenizer)
             attention_mask = torch.ones_like(input_ids)
        
        # Expand attention mask to match hidden state dimensions: (batch, seq_len) -> (batch, seq_len, 1)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # Sum embeddings of valid tokens (zeros out padding)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # Sum mask to get count of valid tokens (avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Calculate mean
        mean_pooled_output = sum_embeddings / sum_mask
        # --- End Pooling Calculation ---

        # Concatenate CLS and Mean Pooled outputs
        combined_features = torch.cat((cls_hidden_state, mean_pooled_output), dim=1)

        # Apply dropout (applied once to the combined features before the head)
        pooled_output = self.bert_output_dropout(combined_features)
        
        # Pass through the advanced classifier head
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