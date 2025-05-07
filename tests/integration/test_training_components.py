import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
import os
import yaml
from pathlib import Path
from transformers import ModernBertConfig

# Functions/classes from src to test
from src.models import ModernBertForSentiment
from src.train import load_config, train # train function for checkpointing test primarily
from src.evaluation import evaluate # Assuming evaluate is in src.evaluation
from src.train_utils import SentimentWeightedLoss # As an example

# Fixtures are imported from tests.integration.conftest (tiny_bert_config, tiny_model, dummy_dataloader etc.)

class TestTrainingComponentsIntegration:

    def test_model_initialization_and_forward_pass(self, tiny_modern_bert_sentiment_model, dummy_integration_dataloader):
        model = tiny_modern_bert_sentiment_model
        batch = next(iter(dummy_integration_dataloader))

        # Ensure model is on CPU for this test unless GPU is specifically tested elsewhere
        model.to(torch.device("cpu")) 

        outputs = model(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            lengths=batch['lengths']
        )

        assert outputs.loss is not None
        assert isinstance(outputs.loss, torch.Tensor)
        assert outputs.logits is not None
        assert outputs.logits.shape == (dummy_integration_dataloader.batch_size, model.config.num_labels)

    def test_single_training_step(self, tiny_modern_bert_sentiment_model, dummy_integration_dataloader):
        model = tiny_modern_bert_sentiment_model
        optimizer = AdamW(model.parameters(), lr=1e-3)
        # Use a real loss function instance from train_utils
        # The model already instantiates its own loss_fct based on config.

        model.train() # Set to training mode
        batch = next(iter(dummy_integration_dataloader))

        original_params = {name: p.clone() for name, p in model.named_parameters() if p.requires_grad}

        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            lengths=batch['lengths']
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        assert loss.item() > 0 # Simplistic check, assumes loss is not zero
        # Check if parameters have been updated
        for name, p_orig in original_params.items():
            p_updated = dict(model.named_parameters())[name]
            assert p_updated.grad is not None # Gradients should exist
            assert not torch.allclose(p_orig, p_updated), f"Parameter {name} was not updated"

    def test_evaluation_flow(self, tiny_modern_bert_sentiment_model, dummy_integration_dataloader):
        model = tiny_modern_bert_sentiment_model
        # Ensure model is in eval mode for evaluation function
        model.eval() 

        # The evaluate function from src.evaluation is expected to take model, dataloader, device
        # For this integration test, we'll run on CPU.
        device = torch.device("cpu")
        model.to(device)

        # Note: The dummy_integration_dataloader's labels are float. 
        # The evaluate function might expect integer labels for accuracy/F1 sklearn metrics.
        # The current `evaluate` in `src/evaluation.py` handles sigmoid on logits and thresholding.
        # It should be fine with float labels that are 0.0 or 1.0.

        metrics = evaluate(model, dummy_integration_dataloader, device)

        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert isinstance(metrics['accuracy'], float)
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert isinstance(metrics['f1'], float)
        assert 0.0 <= metrics['f1'] <= 1.0

    def test_checkpoint_save_and_load_integration(self, tiny_modern_bert_sentiment_model, dummy_integration_dataloader, test_config_for_components, tmp_path):
        """
        This test simulates a minimal training loop part of the `train` function from `src.train`,
        specifically focusing on saving and then loading a checkpoint.
        It verifies that model weights, optimizer state, and other metadata are correctly handled.
        """
        device = torch.device("cpu")
        model = tiny_modern_bert_sentiment_model.to(device)
        optimizer = AdamW(model.parameters(), lr=float(test_config_for_components['training']['lr']))
        # Dummy scheduler for completeness, though its state might not be deeply checked here
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        # --- Simulate part of the training process to get some state ---
        model.train()
        batch = next(iter(dummy_integration_dataloader))
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], lengths=batch['lengths'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch = 0
        best_f1 = 0.8 # Dummy value
        # Use the output_dir from the test_config_for_components, which is a tmp_path
        output_dir = Path(test_config_for_components['model']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"checkpoint_epoch{epoch}_f1{best_f1:.4f}.pth"

        # --- Save checkpoint (logic adapted from train.py) ---
        checkpoint_content_to_save = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
            'config': test_config_for_components # Save the config used for this "training"
        }
        torch.save(checkpoint_content_to_save, checkpoint_path)
        assert checkpoint_path.exists()

        # --- Re-initialize model, optimizer, scheduler (as if starting a new run) ---
        # Use the same config for the new model that was saved in the checkpoint
        config_from_checkpoint_for_new_model = checkpoint_content_to_save['config']['model']
        # Create a new ModernBertConfig instance from the dictionary
        new_bert_config = ModernBertConfig(**{k: v for k, v in tiny_modern_bert_sentiment_model.config.to_dict().items() if k not in ['loss_function']}) # Base attributes
        new_bert_config.pooling_strategy = config_from_checkpoint_for_new_model['pooling_strategy']
        new_bert_config.num_weighted_layers = config_from_checkpoint_for_new_model['num_weighted_layers']
        new_bert_config.loss_function = config_from_checkpoint_for_new_model['loss_function']
        new_bert_config.classifier_dropout = config_from_checkpoint_for_new_model.get('dropout', 0.1)
        new_bert_config.num_labels = 1 # Ensure this matches

        new_model = ModernBertForSentiment(config=new_bert_config).to(device)
        new_optimizer = AdamW(new_model.parameters(), lr=float(test_config_for_components['training']['lr']))
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.1)

        # --- Load checkpoint (logic adapted from train.py) ---
        loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        new_scheduler.load_state_dict(loaded_checkpoint['scheduler_state_dict'])
        loaded_epoch = loaded_checkpoint['epoch']
        loaded_best_f1 = loaded_checkpoint['best_f1']
        loaded_config = loaded_checkpoint['config']

        # --- Verifications ---
        assert loaded_epoch == epoch
        assert loaded_best_f1 == best_f1
        assert loaded_config['model']['name'] == test_config_for_components['model']['name']

        # Verify model parameters are the same
        for param_name, param_loaded in new_model.named_parameters():
            original_param = model.state_dict()[param_name]
            assert torch.allclose(param_loaded, original_param), f"Model param {param_name} mismatch after loading"

        # Verify optimizer state (this is a bit more complex, check a few things)
        # A simple check: compare a parameter's state in the optimizer if possible
        # For AdamW, state includes 'step', 'exp_avg', 'exp_avg_sq'
        # This check is highly dependent on the optimizer's internal state structure.
        # A less brittle check might be to ensure optimizer.load_state_dict didn't error
        # and a subsequent step behaves as expected (harder to isolate here).
        # For now, just check that states are loaded and not empty if they exist.
        if new_optimizer.state_dict()['state']:
            assert len(new_optimizer.state_dict()['state']) == len(optimizer.state_dict()['state'])

        # Verify a subsequent forward pass gives same results (with eval mode for determinism)
        model.eval()
        new_model.eval()
        batch_for_check = next(iter(dummy_integration_dataloader))
        with torch.no_grad():
            outputs_original = model(input_ids=batch_for_check['input_ids'], attention_mask=batch_for_check['attention_mask'])
            outputs_loaded = new_model(input_ids=batch_for_check['input_ids'], attention_mask=batch_for_check['attention_mask'])
        assert torch.allclose(outputs_loaded.logits, outputs_original.logits, atol=1e-6), "Logits mismatch after loading checkpoint"
