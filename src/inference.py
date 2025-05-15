import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.models import ModernBertForSentiment
from transformers import ModernBertConfig
from typing import Dict, Any
import yaml
import os
from torch.cuda.amp import autocast

use_cuda = torch.cuda.is_available()

class SentimentInference:
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration and initialize model and tokenizer."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_cfg = config.get('model', {})
        inference_cfg = config.get('inference', {})

        if use_cuda:
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Path to the .pt model weights file
        model_weights_path = inference_cfg.get('model_path', 
                                             os.path.join(model_cfg.get('output_dir', 'checkpoints'), 'best_model.pt'))
        
        # Base model name from config (e.g., 'answerdotai/ModernBERT-base')
        # This will be used for loading both tokenizer and base BERT config from Hugging Face Hub
        base_model_name = model_cfg.get('name', 'answerdotai/ModernBERT-base')

        self.max_length = inference_cfg.get('max_length', model_cfg.get('max_length', 256))

        # Load tokenizer from the base model name (e.g., from Hugging Face Hub)
        print(f"Loading tokenizer from: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base BERT config from the base model name
        print(f"Loading ModernBertConfig from: {base_model_name}")
        bert_config = ModernBertConfig.from_pretrained(base_model_name) 
        
        # --- Apply any necessary overrides from your config to the loaded bert_config --- 
        # For example, if your ModernBertForSentiment expects specific config values beyond the base BERT model.
        # Your current ModernBertForSentiment takes the entire config object, which might implicitly carry these.
        # However, explicitly setting them on bert_config loaded from HF is safer if they are architecturally relevant.
        bert_config.classifier_dropout = model_cfg.get('dropout', bert_config.classifier_dropout) # Example
        # Ensure num_labels is set if your inference model needs it (usually for HF pipeline, less so for manual predict)
        # bert_config.num_labels = model_cfg.get('num_labels', 1) # Typically 1 for binary sentiment regression-style output

        # It's also important that pooling_strategy and num_weighted_layers are set on the config object 
        # that ModernBertForSentiment receives, as it uses these to build its layers.
        # These are usually fine-tuning specific, not part of the base HF config, so they should come from your model_cfg.
        bert_config.pooling_strategy = model_cfg.get('pooling_strategy', 'cls')
        bert_config.num_weighted_layers = model_cfg.get('num_weighted_layers', 4)
        bert_config.loss_function = model_cfg.get('loss_function', {'name': 'SentimentWeightedLoss', 'params': {}}) # Needed by model init
        # Ensure num_labels is explicitly set for the model's classifier head
        bert_config.num_labels = 1 # For sentiment (positive/negative) often treated as 1 logit output

        print("Instantiating ModernBertForSentiment model structure...")
        self.model = ModernBertForSentiment(bert_config)
        
        print(f"Loading model weights from local checkpoint: {model_weights_path}")
        # Load the entire checkpoint dictionary first
        checkpoint = torch.load(model_weights_path, map_location=self.device)
        
        # Extract the model_state_dict from the checkpoint
        # This handles the case where the checkpoint saves more than just the model weights (e.g., optimizer state, epoch)
        if 'model_state_dict' in checkpoint:
            model_state_to_load = checkpoint['model_state_dict']
        else:
            # If the checkpoint is just the state_dict itself (older format or different saving convention)
            model_state_to_load = checkpoint
            
        self.model.load_state_dict(model_state_to_load)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")
        
    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True)
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        with torch.no_grad():
            if use_cuda:
                with autocast():
                    outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            else:
                outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs["logits"]
        prob = torch.sigmoid(logits).item()
        return {"sentiment": "positive" if prob > 0.5 else "negative", "confidence": prob}