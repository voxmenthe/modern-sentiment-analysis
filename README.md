# Modern Sentiment Analysis with ModernBERT

This script trains a ModernBERT‑based sentiment‑classification model on the IMDB dataset.  Key components:

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd modern-sentiment-analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure hyperparameters (optional):**
    Modify `src/config.yaml` to adjust model name, training parameters (epochs, batch size, learning rate), max sequence length, or output directory.

## Usage

### Training

Run the training script from the root directory:

```bash
python src/train.py
```

-   The script will load configuration from `src/config.yaml`.
-   It downloads the IMDB dataset.
-   Trains the ModernBERT model for sentiment classification.
-   Saves the best performing model (based on F1 score on the validation set) to the `output_dir` specified in the config (default: `checkpoints/best_model.pt`).

### Inference

You can use the trained model for inference on new text. (Example script/usage to be added if needed).

```python
# Example usage within a Python script
from src.inference import SentimentInference

# Assumes config.yaml points to the correct trained model path
inferer = SentimentInference() 

text_positive = "This movie was fantastic! Highly recommended."
text_negative = "A truly disappointing and boring film."

result_pos = inferer.predict(text_positive)
result_neg = inferer.predict(text_negative)

print(f'"{text_positive}" -> {result_pos}')
print(f'"{text_negative}" -> {result_neg}')
```

# Usage (minimal):
#   python train.py --model_name modernbert-base --epochs 3
# --------------------------------------------------------------