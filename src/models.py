from transformers import ModernBertModel, ModernBertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch
from train_utils import SentimentWeightedLoss
import torch.nn.functional as F

from classifiers import EnhancedClassifierHead, AdvancedClassifierHead, ConcatEnhancedClassifierHead


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
        
        # Use the new CONCATENATED enhanced classifier head
        # Input size is doubled because we concatenate CLS and Mean Pool outputs
        self.classifier = ConcatEnhancedClassifierHead(
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