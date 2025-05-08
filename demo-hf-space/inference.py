import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, ModernBertConfig
from typing import Dict, Any
import yaml
import os
from models import ModernBertForSentiment

class SentimentInference:
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration and initialize model and tokenizer from local checkpoint or Hugging Face Hub."""
        print(f"--- Debug: SentimentInference __init__ received config_path: {config_path} ---") # Add this
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        print(f"--- Debug: SentimentInference loaded config_data: {config_data} ---") # Add this
        
        model_yaml_cfg = config_data.get('model', {})
        inference_yaml_cfg = config_data.get('inference', {})
        
        # Determine device early
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available(): # Check for MPS (Apple Silicon GPU)
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"[INFERENCE_LOG] Using device: {self.device}")
        
        model_hf_repo_id = model_yaml_cfg.get('name_or_path')
        tokenizer_hf_repo_id = model_yaml_cfg.get('tokenizer_name_or_path', model_hf_repo_id)
        local_model_weights_path = inference_yaml_cfg.get('model_path') # Path for local .pt file

        print(f"--- Debug: model_hf_repo_id: {model_hf_repo_id} ---") # Add this
        print(f"--- Debug: local_model_weights_path: {local_model_weights_path} ---") # Add this

        self.max_length = inference_yaml_cfg.get('max_length', model_yaml_cfg.get('max_length', 512))

        # --- Tokenizer Loading (always from Hub for now, or could be made conditional) ---
        if not tokenizer_hf_repo_id and not model_hf_repo_id:
            raise ValueError("Either model.tokenizer_name_or_path or model.name_or_path (as fallback for tokenizer) must be specified in config.yaml")
        effective_tokenizer_repo_id = tokenizer_hf_repo_id or model_hf_repo_id
        print(f"[INFERENCE_LOG] Loading tokenizer from: {effective_tokenizer_repo_id}") # Logging
        self.tokenizer = AutoTokenizer.from_pretrained(effective_tokenizer_repo_id)

        # --- Model Loading --- #
        # Determine if we are loading from a local .pt file or from Hugging Face Hub
        load_from_local_pt = False
        if local_model_weights_path and os.path.isfile(local_model_weights_path):
            print(f"[INFERENCE_LOG] Found local model weights path: {local_model_weights_path}") # Logging
            print(f"--- Debug: Found local model weights path: {local_model_weights_path} ---") # Add this
            load_from_local_pt = True
        elif not model_hf_repo_id:
            raise ValueError("No local model_path found and model.name_or_path (for Hub) is not specified in config.yaml")

        print(f"[INFERENCE_LOG] load_from_local_pt: {load_from_local_pt}") # Logging
        print(f"--- Debug: load_from_local_pt is: {load_from_local_pt} ---") # Add this

        if load_from_local_pt:
            print("[INFERENCE_LOG] Attempting to load model from LOCAL .pt checkpoint...") # Logging
            print("--- Debug: Entering LOCAL .pt loading path ---") # Add this
            # Base BERT config must still be loaded, usually from a Hub ID (e.g., original base model)
            # This base_model_for_config_id is crucial for building the correct ModernBertForSentiment structure.
            base_model_for_config_id = model_yaml_cfg.get('base_model_for_config', model_yaml_cfg.get('name_or_path'))
            if not base_model_for_config_id:
                 raise ValueError("model.base_model_for_config or model.name_or_path must be specified in config.yaml when loading local .pt for ModernBertForSentiment structure.")
            
            print(f"[INFERENCE_LOG] LOCAL_PT_LOAD: base_model_for_config_id: {base_model_for_config_id}") # Logging

            model_config = ModernBertConfig.from_pretrained(
                base_model_for_config_id, 
                num_labels=model_yaml_cfg.get('num_labels', 1), # from config.yaml via model_yaml_cfg
                pooling_strategy=model_yaml_cfg.get('pooling_strategy', 'mean'), # from config.yaml via model_yaml_cfg
                num_weighted_layers=model_yaml_cfg.get('num_weighted_layers', 4) # from config.yaml via model_yaml_cfg
            )
            print(f"[INFERENCE_LOG] LOCAL_PT_LOAD: Loaded ModernBertConfig: {model_config.to_diff_dict()}") # Logging

            print(f"[INFERENCE_LOG] LOCAL_PT_LOAD: Initializing ModernBertForSentiment with this config.") # Logging
            self.model = ModernBertForSentiment(config=model_config)
            
            print(f"[INFERENCE_LOG] LOCAL_PT_LOAD: Loading weights from checkpoint: {local_model_weights_path}") # Logging
            checkpoint = torch.load(local_model_weights_path, map_location=torch.device('cpu'))
            
            state_dict_to_load = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            if not isinstance(state_dict_to_load, dict):
                raise TypeError(f"Loaded checkpoint from {local_model_weights_path} is not a dict or does not contain 'model_state_dict' or 'state_dict'.")

            # Log first few keys for debugging
            first_few_keys = list(state_dict_to_load.keys())[:5]
            print(f"[INFERENCE_LOG] LOCAL_PT_LOAD: First few keys from checkpoint state_dict: {first_few_keys}") # Logging

            self.model.load_state_dict(state_dict_to_load)
            print(f"[INFERENCE_LOG] LOCAL_PT_LOAD: Weights loaded successfully into ModernBertForSentiment from {local_model_weights_path}.") # Logging
        else:
            # Load from Hugging Face Hub
            print(f"[INFERENCE_LOG] Attempting to load model from HUGGING_FACE_HUB: {model_hf_repo_id}") # Logging
            
            hub_config_params = {
                "num_labels": model_yaml_cfg.get('num_labels', 1),
                "pooling_strategy": model_yaml_cfg.get('pooling_strategy', 'mean'),
                "num_weighted_layers": model_yaml_cfg.get('num_weighted_layers', 6)
            }
            print(f"[INFERENCE_LOG] HUB_LOAD: Parameters to update Hub config: {hub_config_params}") # Logging

            try:
                # Step 1: Load config from Hub, allowing for our custom ModernBertConfig
                config = ModernBertConfig.from_pretrained(model_hf_repo_id)
                # Step 2: Update the loaded config with our specific parameters
                for key, value in hub_config_params.items():
                    setattr(config, key, value)
                print(f"[INFERENCE_LOG] HUB_LOAD: Updated config: {config.to_diff_dict()}")

                # Step 3: Load model with the updated config
                self.model = ModernBertForSentiment.from_pretrained(
                    model_hf_repo_id,
                    config=config
                )
                print(f"[INFERENCE_LOG] HUB_LOAD: Model ModernBertForSentiment loaded successfully from {model_hf_repo_id} with updated config.") # Logging
            except Exception as e:
                print(f"[INFERENCE_LOG] HUB_LOAD: Error loading ModernBertForSentiment from {model_hf_repo_id} with explicit config: {e}") # Logging
                print(f"[INFERENCE_LOG] HUB_LOAD: Falling back to AutoModelForSequenceClassification for {model_hf_repo_id}.") # Logging
                
                # Fallback: Try with AutoModelForSequenceClassification
                # Load its config (could be BertConfig or ModernBertConfig if auto-detected)
                # AutoConfig should ideally resolve to ModernBertConfig if architectures field is set in Hub's config.json
                try:
                    config_fallback = AutoConfig.from_pretrained(model_hf_repo_id)
                    for key, value in hub_config_params.items():
                        setattr(config_fallback, key, value)
                    print(f"[INFERENCE_LOG] HUB_LOAD_FALLBACK: Updated fallback config: {config_fallback.to_diff_dict()}")

                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_hf_repo_id,
                        config=config_fallback
                    )
                    print(f"[INFERENCE_LOG] HUB_LOAD_FALLBACK: AutoModelForSequenceClassification loaded for {model_hf_repo_id} with updated config.") # Logging
                except Exception as e_fallback:
                    print(f"[INFERENCE_LOG] HUB_LOAD_FALLBACK: Critical error during fallback load: {e_fallback}")
                    raise e_fallback # Re-raise if fallback also fails catastrophically

        self.model.to(self.device) # Move model to the determined device
        self.model.eval()
        
    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs['input_ids'].to(self.device), attention_mask=inputs['attention_mask'].to(self.device))
        logits = outputs.get("logits") # Use .get for safety
        if logits is None:
            raise ValueError("Model output did not contain 'logits'. Check model's forward pass.")
        prob = torch.sigmoid(logits).item()
        return {"sentiment": "positive" if prob > 0.5 else "negative", "confidence": prob}