# tests/conftest.py
import sys
import os
from huggingface_hub import login

print("DEBUG: Executing tests/conftest.py")


hf_token = os.getenv("HF_TOKEN")
if hf_token:
    try:
        login(token=hf_token)
        print("Successfully logged into Hugging Face Hub programmatically.")
    except Exception as e:
        print(f"Failed to log into Hugging Face Hub programmatically: {e}")
else:
    print("HF_TOKEN environment variable not found. Proceeding without programmatic login.")

# Get the absolute path to the current project's root (mps_debottleneck_training)
# Assuming conftest.py is in WORKSPACE_ROOT/tests/
current_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if current_project_root not in sys.path:
    sys.path.insert(0, current_project_root)
    print(f"DEBUG: tests/conftest.py: Prepended to sys.path: {current_project_root}")

print(f"DEBUG: tests/conftest.py: sys.path is now: {sys.path}") 