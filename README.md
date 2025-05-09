# Modern Sentiment Analysis with ModernBERT

This script trains a ModernBERT‑based sentiment‑classification model on the IMDB dataset.  

## Key features
* Custom loss functions (described in `LOSS_FUNCTION_THOUGHTS.md`)
* The use of a custom classification head with skip connections
* Building on top of the pre-trained ModernBERT model.

## Key components
* `src/train.py`: Main training script.
* `src/config.yaml`: Configuration file.
* `src/model.py`: Model definition.
* `src/data_utils.py`: Data loading and preprocessing.
* `src/train_utils.py`: The custom loss functions are defined here.
* `src/classifiers.py`: The custom classification head is defined here.
* `src/inference.py`: Inference script.

## Demo Space

There is a simple web demo space hosted on Hugging Face Spaces at this link that uses the trained model created with this repo:

[Hugging Face Demo Space](https://huggingface.co/spaces/voxmenthe/imdb-sentiment-demo)


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

## Potential Improvements

**Data Augmentation:** One area that I typically spend much more time on is data augmentation. This often has a much higher ROI than model architecture changes, and even custom loss functions, but was not in the scope of this project. A few ideas:
* Augment the data by changing some of the vocabulary and phrasing. Beyond simple heuristics, LLMs could be used to generate new samples, or even just to change the phrasing of the existing samples.
* Split samples above a certain size into multiple samples (same label). This could get really fancy with semantic chunking, but that's probably overkill.
* Truncation is already implemented, but could be improved.

**Model Architecture:** Maybe not the highest ROI area, but there are a few things I'd like to try:
* Ensembling could be used to improve performance.
* Numberous other architectures could be tried.
* The classification head could probably be improved from my current two-layer ResNet-like structure.

**Training Optimizations:** There's a lot of low-hanging fruit here that I didn't have a chance to get to.
* Some kind of averaging strategy such as Stochastic Weight Averaging (SWA) or Model Soups could be used to improve performance.
* I'd like to experiment with more sophisticated learning rate scheduling.
* I just used AdamW, but a more sophisticated optimizer such as Muon could be used.
* And much more.