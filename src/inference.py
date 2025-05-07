import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models import ModernBertForSentiment
from transformers import ModernBertConfig
from typing import Dict, Any
import yaml


class SentimentInference:
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration and initialize model and tokenizer."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_cfg = config.get('model', {})
        inference_cfg = config.get('inference', {})
        
        # Use inference model path if specified, else default from training output
        model_path = inference_cfg.get('model_path', model_cfg.get('output_dir', 'checkpoints') + '/best_model.pt')
        self.max_length = inference_cfg.get('max_length', model_cfg.get('max_length', 256))

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        bert_config = ModernBertConfig.from_pretrained(model_cfg.get('name', 'answerdotai/ModernBERT-base')) 
        self.model = ModernBertForSentiment(bert_config)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs["logits"]
        prob = torch.sigmoid(logits).item()
        return {"sentiment": "positive" if prob > 0.5 else "negative", "confidence": prob}