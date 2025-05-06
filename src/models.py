from transformers import ModernBertModel, ModernBertPreTrainedModel
from torch import nn
from train_utils import SentimentWeightedLoss


class ModernBertForSentiment(ModernBertPreTrainedModel):
    """ModernBERT encoder with a classification head (single logit)."""

    def __init__(self, config):
        super().__init__(config)
        self.bert = ModernBertModel(config)  # Define the base model attribute
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights correctly, especially for the new classifier head
        # self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        lengths=None,
    ):
        outputs = self.bert(  # Use self.bert again
            input_ids,
            attention_mask=attention_mask,
        )
        # Use the hidden state of the [CLS] token
        cls_hidden_state = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(cls_hidden_state)
        logits = self.classifier(pooled).squeeze(-1)  # [B]
        loss = None
        if labels is not None and lengths is not None:
            loss_fct = SentimentWeightedLoss()
            loss = loss_fct(logits, labels, lengths)
        return {"loss": loss, "logits": logits}