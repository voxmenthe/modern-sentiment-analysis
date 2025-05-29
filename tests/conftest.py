import sys
import os

# Add the project root directory to sys.path
# This allows imports like `from src.module import ...` from within tests
# and also helps if modules within `src` use absolute `src.` imports.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) 