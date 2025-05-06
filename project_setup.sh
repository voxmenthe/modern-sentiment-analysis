#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Upgrade pip and install poetry
pip install --upgrade pip
pip install poetry

# Update the lock file if necessary
poetry lock

# Install dependencies and the project
poetry install

# Create and install the IPython kernel for the project
python -m ipykernel install --user --name=modern-sentiment-analysis --display-name "Modern Sentiment Analysis" # install globally outside of poetry

echo "Jupyter kernel 'modern-sentiment-analysis' has been installed."

echo "Project setup complete!"