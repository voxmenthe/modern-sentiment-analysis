from huggingface_hub import HfApi, upload_folder, create_repo, login
from transformers import AutoTokenizer, AutoConfig
import os
import shutil
import tempfile
import torch
import argparse

# --- Configuration ---
HUGGING_FACE_USERNAME = "voxmenthe"  # Your Hugging Face username
MODEL_NAME_ON_HF = "modernbert-imdb-sentiment" # The name of the model on Hugging Face
REPO_ID = f"{HUGGING_FACE_USERNAME}/{MODEL_NAME_ON_HF}"

# Original base model from which the tokenizer and initial config were derived
ORIGINAL_BASE_MODEL_NAME = "answerdotai/ModernBERT-base"

# Local path to your fine-tuned model checkpoint
LOCAL_MODEL_CHECKPOINT_DIR = "checkpoints"
FINE_TUNED_MODEL_FILENAME = "mean_epoch5_0.9575acc_0.9575f1.pt" # Your best checkpoint
# If your fine-tuned model is just a .pt file, ensure you also have a config.json for ModernBert
# For simplicity, we'll re-save the config from the fine-tuned model structure if possible, or from original base.

# Files from your project to include (e.g., custom model code, inference script)
# The user has moved these to the root directory.
PROJECT_FILES_TO_UPLOAD = [
    "config.yaml",
    "inference.py",
    "models.py",
    "train_utils.py",
    "classifiers.py",
    "README.md"
]

def upload_model_and_tokenizer():
    api = HfApi()

    REPO_ID = f"{HUGGING_FACE_USERNAME}/{MODEL_NAME_ON_HF}"
    print(f"Preparing to upload to Hugging Face Hub repository: {REPO_ID}")

    # Create the repository on Hugging Face Hub if it doesn't exist
    # This should be done after login to ensure correct permissions
    print(f"Ensuring repository '{REPO_ID}' exists on Hugging Face Hub...")
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repository '{REPO_ID}' ensured.")
    except Exception as e:
        print(f"Error creating/accessing repository {REPO_ID}: {e}")
        print("Please check your Hugging Face token and repository permissions.")
        return

    # Create a temporary directory to gather all files for upload
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory for upload: {temp_dir}")

        # 1. Save tokenizer files from the ORIGINAL_BASE_MODEL_NAME
        print(f"Saving tokenizer from {ORIGINAL_BASE_MODEL_NAME} to {temp_dir}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_BASE_MODEL_NAME)
            tokenizer.save_pretrained(temp_dir)
            print("Tokenizer files saved.")
        except Exception as e:
            print(f"Error saving tokenizer from {ORIGINAL_BASE_MODEL_NAME}: {e}")
            print("Please ensure this model name is correct and accessible.")
            return

        # 2. Save base model config.json (architecture) from ORIGINAL_BASE_MODEL_NAME
        # This is crucial for AutoModelForSequenceClassification.from_pretrained(REPO_ID) to work.
        print(f"Saving model config.json from {ORIGINAL_BASE_MODEL_NAME} to {temp_dir}...")
        try:
            config = AutoConfig.from_pretrained(ORIGINAL_BASE_MODEL_NAME)
            print(f"Config loaded. Initial num_labels (if exists): {getattr(config, 'num_labels', 'Not set')}")
            
            # Set architecture first
            config.architectures = ["ModernBertForSentiment"]

            # Add necessary classification head attributes for AutoModelForSequenceClassification
            config.num_labels = 1 # For IMDB sentiment (binary, single logit output based on training)
            print(f"After attempting to set: config.num_labels = {config.num_labels}")
            
            config.id2label = {0: "NEGATIVE", 1: "POSITIVE"} # Standard for binary, even with num_labels=1
            config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
            print(f"After setting id2label/label2id, config.num_labels is: {config.num_labels}")

            # CRITICAL: Force num_labels to 1 again immediately before saving
            config.num_labels = 1
            print(f"Immediately before save, FINAL check config.num_labels = {config.num_labels}")
            
            # Safeguard: Remove any existing config.json from temp_dir before saving ours
            potential_old_config_path = os.path.join(temp_dir, "config.json")
            if os.path.exists(potential_old_config_path):
                os.remove(potential_old_config_path)
                print(f"Removed existing config.json from {temp_dir} to ensure clean save.")

            config.save_pretrained(temp_dir)
            print(f"Model config.json (with num_labels={config.num_labels}, architectures={config.architectures}) saved to {temp_dir}.")
        except Exception as e:
            print(f"Error saving config.json from {ORIGINAL_BASE_MODEL_NAME}: {e}")
            return

        # Load the fine-tuned model checkpoint to extract the state_dict
        full_checkpoint_path = os.path.join(LOCAL_MODEL_CHECKPOINT_DIR, FINE_TUNED_MODEL_FILENAME)
        hf_model_path = os.path.join(temp_dir, "pytorch_model.bin")

        if not os.path.exists(full_checkpoint_path):
            print(f"ERROR: Local model checkpoint not found at {full_checkpoint_path}")
            shutil.rmtree(temp_dir)
            return

        print(f"Loading local checkpoint from: {full_checkpoint_path}")
        # Load checkpoint to CPU to avoid GPU memory issues if the script runner doesn't have/need GPU
        checkpoint = torch.load(full_checkpoint_path, map_location='cpu')

        model_state_dict = None
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print("Extracted 'model_state_dict' from checkpoint.")
        elif 'state_dict' in checkpoint: # Another common key for state_dicts
            model_state_dict = checkpoint['state_dict']
            print("Extracted 'state_dict' from checkpoint.")
        elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
            # If the checkpoint is already a state_dict (e.g., from torch.save(model.state_dict(), ...))
            # Basic check: does it have keys that look like weights/biases?
            if any(key.endswith('.weight') or key.endswith('.bias') for key in checkpoint.keys()):
                model_state_dict = checkpoint
                print("Checkpoint appears to be a raw state_dict (contains .weight or .bias keys).")
            else:
                print("Checkpoint is a dict, but does not immediately appear to be a state_dict (no .weight/.bias keys found).")
                print(f"Checkpoint keys: {list(checkpoint.keys())[:10]}...") # Print some keys for diagnosis

        else:
            # This case handles if checkpoint is not a dict or doesn't match known structures
            print(f"ERROR: Could not find a known state_dict key in the checkpoint, and it's not a recognizable raw state_dict.")
            if isinstance(checkpoint, dict):
                print(f"Checkpoint dictionary keys found: {list(checkpoint.keys())}")
            else:
                print(f"Checkpoint is not a dictionary. Type: {type(checkpoint)}")
            shutil.rmtree(temp_dir)
            return

        if model_state_dict is None:
            print("ERROR: model_state_dict was not successfully extracted. Aborting upload.")
            shutil.rmtree(temp_dir)
            return

        # --- DEBUG: Print keys of the state_dict --- 
        print("\n--- Keys in extracted (original) model_state_dict (first 30 and last 10): ---")
        state_dict_keys = list(model_state_dict.keys())
        if len(state_dict_keys) > 0:
            for i, key in enumerate(state_dict_keys[:30]):
                print(f"  {i+1}. {key}")
            if len(state_dict_keys) > 40: # Show ellipsis if there's a gap
                print("  ...")
            # Print last 10 keys if there are more than 30
            start_index_for_last_10 = max(30, len(state_dict_keys) - 10)
            for i, key_idx in enumerate(range(start_index_for_last_10, len(state_dict_keys))):
                print(f"  {key_idx+1}. {state_dict_keys[key_idx]}")
        else:
            print("  (No keys found in model_state_dict)")
        print(f"Total keys: {len(state_dict_keys)}")
        print("-----------------------------------------------------------\n")
        # --- END DEBUG --- 

        # Transform keys for Hugging Face compatibility if needed.
        # For ModernBertForSentiment with self.bert and self.classifier (custom head):
        # - Checkpoint 'bert.*' should remain 'bert.*'
        # - Checkpoint 'classifier.*' keys (e.g., classifier.dense1.weight, classifier.out_proj.weight) should remain 'classifier.*' as they are.
        transformed_state_dict = {}
        has_classifier_weights_transformed = False # Used to track if out_proj was found

        print("Transforming state_dict keys for Hugging Face Hub compatibility...")
        for key, value in model_state_dict.items():
            new_key = None
            if key.startswith("bert."):
                # Keep 'bert.' prefix as ModernBertForSentiment uses self.bert
                new_key = key 
            elif key.startswith("classifier."):
                # All parts of the custom classifier head should retain their names
                new_key = key
                if "out_proj" in key: # Just to confirm it exists
                     has_classifier_weights_transformed = True # Indicate out_proj was found and processed
        
            if new_key:
                transformed_state_dict[new_key] = value
                if key != new_key:
                    print(f"  Mapping '{key}' -> '{new_key}'")
                else:
                    # print(f"  Keeping key as is: '{key}'") # Optional
                    pass 
            else:
                print(f"  INFO: Discarding key not mapped: {key}")

        # Check if the critical classifier output layer was present in the source checkpoint
        # This check might need adjustment based on the actual layers of ClassifierHead
        # For now, we check if any 'out_proj' key was seen under 'classifier.'
        if not has_classifier_weights_transformed:
            print("WARNING: No 'classifier.out_proj.*' keys were found in the source checkpoint.")
            print("         Ensure your checkpoint contains the expected classifier layers.")
            # Not necessarily an error to abort, as other classifier keys might be valid.

        model_state_dict = transformed_state_dict

        # --- DEBUG: Print keys of the TRANSFORMED state_dict ---        
        print("\n--- Keys in TRANSFORMED model_state_dict for upload (first 30 and last 10): ---")
        state_dict_keys_transformed = list(transformed_state_dict.keys())
        if len(state_dict_keys_transformed) > 0:
            for i, key_t in enumerate(state_dict_keys_transformed[:30]):
                print(f"  {i+1}. {key_t}")
            if len(state_dict_keys_transformed) > 40:
                print("  ...")
            start_index_for_last_10_t = max(30, len(state_dict_keys_transformed) - 10)
            for i, key_idx_t in enumerate(range(start_index_for_last_10_t, len(state_dict_keys_transformed))):
                print(f"  {key_idx_t+1}. {state_dict_keys_transformed[key_idx_t]}")
        else:
            print("  (No keys found in transformed_state_dict)")
        print(f"Total keys in transformed_state_dict: {len(state_dict_keys_transformed)}")
        print("-----------------------------------------------------------\n")

        # Save the TRANSFORMED state_dict
        torch.save(transformed_state_dict, hf_model_path)
        print(f"Saved TRANSFORMED model state_dict to {hf_model_path}.")

        # 4. Copy other project files
        for project_file in PROJECT_FILES_TO_UPLOAD:
            local_project_file_path = project_file # Files are now at the root
            if os.path.exists(local_project_file_path):
                shutil.copy(local_project_file_path, os.path.join(temp_dir, os.path.basename(project_file)))
                print(f"Copied project file {project_file} to {temp_dir}.")

        # Before uploading, let's inspect the temp_dir to be absolutely sure what's there
        print(f"--- Inspecting temp_dir ({temp_dir}) before upload: ---")
        for item in os.listdir(temp_dir):
            print(f"  - {item}")
        temp_config_path_to_check = os.path.join(temp_dir, "config.json")
        if os.path.exists(temp_config_path_to_check):
            print(f"--- Content of {temp_config_path_to_check} before upload: ---")
            with open(temp_config_path_to_check, 'r') as f_check:
                print(f_check.read())
            print("--- End of config.json content ---")
        else:
            print(f"WARNING: {temp_config_path_to_check} does NOT exist before upload!")

        # 5. Upload the contents of the temporary directory
        print(f"Uploading all files from {temp_dir} to {REPO_ID}...")
        try:
            upload_folder(
                folder_path=temp_dir,
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"Upload fine-tuned model, tokenizer, and supporting files for {MODEL_NAME_ON_HF}"
            )
            print("All files uploaded successfully!")
        except Exception as e:
            print(f"Error uploading files: {e}")
        finally:
            print(f"Cleaning up temporary directory: {temp_dir}")
            # The TemporaryDirectory context manager handles cleanup automatically
            # but an explicit message is good for clarity.

        print("Upload process finished.")

if __name__ == "__main__":
    upload_model_and_tokenizer()