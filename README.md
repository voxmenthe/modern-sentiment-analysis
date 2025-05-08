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

First set your desired configuration in `src/config.yaml` or pass command line arguments to override the yaml file's config values.

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

You can use the trained model for inference on new text. 

For a simple check, run the sample inference script:

```bash
python run_sample_inference.py --num_samples 5
```

This will load the model from the default checkpoint and run inference on a few samples from the IMDB validation set.

For more complex usage, you can use the SentimentInference class directly:

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


### Evaluation

You can evaluate the model on the IMDB test set using the evaluation script:

Adjust the `num_samples` parameter to control how many samples to evaluate on or leave it blank for all of them.

It requires that the trained checkpoint be saved locally (e.g. in the `checkpoints` directory) and defined in the `src/config.yaml` file.

```bash
python run_evaluation.py --config src/config.yaml --num_samples 100
```