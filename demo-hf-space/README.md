---
language: en
tags:
- sentiment-analysis
- modernbert
- imdb
datasets:
- imdb
metrics:
- accuracy
- f1
title: IMDb Sentiment Analyzer
emoji: 🤗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.29.0" # Verify this matches your Gradio version in requirements.txt
app_file: app.py
pinned: false
hf_oauth: false
disable_embedding: false
---

# ModernBERT IMDb Sentiment Analysis Model

## Model Description
Fine-tuned ModernBERT model for sentiment analysis on IMDb movie reviews. Achieves 95.75% accuracy on the test set.

## Usage
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("voxmenthe/modernbert-imdb-sentiment")
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

# Input processing
inputs = tokenizer("This movie was fantastic!", return_tensors="pt")
outputs = model(**inputs)

# Get the predicted class
predicted_class_id = outputs.logits.argmax().item()

# Convert class ID to label
predicted_label = model.config.id2label[predicted_class_id]
print(f"Predicted label: {predicted_label}")
```

## Model Card

### Model Details
- **Model Name**: ModernBERT IMDb Sentiment Analysis
- **Base Model**: answerdotai/ModernBERT-base
- **Task**: Sentiment Analysis
- **Dataset**: IMDb Movie Reviews
- **Training Epochs**: 5

### Model Performance
- **Test Accuracy**: 95.75%
- **Test F1 Score**: 95.75%

### Model Architecture
- **Base Model**: answerdotai/ModernBERT-base
- **Task-Specific Head**: ClassifierHead (from `classifiers.py`)
- **Number of Labels**: 2 (Positive, Negative)

### Model Inference
- **Input Format**: Text (single review)
- **Output Format**: Predicted sentiment label (Positive or Negative)

### Model Version
- **Version**: 1.0
- **Date**: 2025-05-07

### Model License
- **License**: MIT License

### Model Contact
- **Contact**: alocalminima@gmail.com

### Model Citation
- **Citation**: voxmenthe/modernbert-imdb-sentiment

## IMDb Sentiment Analyzer - Gradio App

This repository contains a Gradio application for sentiment analysis of IMDb movie reviews.
It is hosted on Hugging Face Spaces at [voxmenthe/imdb-sentiment-demo](https://huggingface.co/spaces/voxmenthe/imdb-sentiment-demo).
It uses a fine-tuned ModernBERT model hosted on Hugging Face.

**Space Link:** [voxmenthe/imdb-sentiment-demo](https://huggingface.co/spaces/voxmenthe/imdb-sentiment-demo)
**Model Link:** [voxmenthe/modernbert-imdb-sentiment](https://huggingface.co/voxmenthe/modernbert-imdb-sentiment)

## Features

*   **Text Input**: Analyze custom movie review text.
*   **Random IMDb Sample**: Load a random review from the IMDb test dataset.
*   **Sentiment Prediction**: Classifies sentiment as Positive or Negative.
*   **True Label Display**: Shows the actual IMDb label for loaded samples.

## Setup & Running Locally

1.  **Clone the repository (or your Space repository):**
    ```bash
    git clone https://huggingface.co/spaces/voxmenthe/imdb-sentiment-demo
    cd imdb-sentiment-demo
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.11+ installed.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python app.py
    ```
    The application will be available at `http://127.0.0.1:7860`.

## Model Information

The sentiment analysis model is a `ModernBERT` architecture fine-tuned on the IMDb dataset. The specific checkpoint used is `mean_epoch5_0.9575acc_0.9575f1.pt` before being uploaded to `voxmenthe/modernbert-imdb-sentiment`.

