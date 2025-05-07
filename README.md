# Modern Sentiment Analysis with ModernBERT

This script trains a ModernBERT‑based sentiment‑classification model on the IMDB dataset.  Key components:

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/voxmenthe/modern-sentiment-analysis.git
    cd modern-sentiment-analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv my_venv
    source my_venv/bin/activate
    ```

3.  **Set up project and install dependencies:**
    ```bash
    ./project_setup.sh
    ```

4.  **Configure hyperparameters (optional):**
    Modify `src/config.yaml` to adjust model name, training parameters (epochs, batch size, learning rate), max sequence length, or output directory.
    You can also pass command line arguments to override the yaml file's config values.

## Usage

### Training

Run the training script from the root directory:

```bash
python src/train.py
```

Other examples:

```bash
python src/train.py --model_name "answerdotai/ModernBERT-base" --epochs 3
python src/train.py --model_name "answerdotai/ModernBERT-large" --epochs 3 --batch_size 16
```

-   The script will load configuration from `src/config.yaml`.
-   It downloads the IMDB dataset.
-   Trains the ModernBERT+Classifier model for sentiment classification.
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
