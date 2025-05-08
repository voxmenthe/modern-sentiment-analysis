import argparse
from datasets import load_dataset
from inference import SentimentInference

def run_sample_inference(config_path: str = "config.yaml", num_samples: int = 5):
    """
    Loads a sentiment analysis model from a checkpoint, runs inference on a few
    samples from the IMDB validation set, and prints the results.
    """
    print("Loading sentiment model...")
    # Initialize SentimentInference
    # Ensure config_path points to your configuration file that specifies the model path
    inferer = SentimentInference(config_path=config_path)
    print("Model loaded.")

    print("\nLoading IMDB dataset (test split for validation samples)...")
    # Load the IMDB dataset, test split is used as validation
    try:
        imdb_dataset = load_dataset("imdb", split="test")
    except Exception as e:
        print(f"Failed to load IMDB dataset: {e}")
        print("Please ensure you have an internet connection and the `datasets` library can access Hugging Face.")
        print("You might need to run `pip install datasets` or check your network settings.")
        return
    
    print(f"Taking {num_samples} samples from the dataset.")
    
    # Take a few samples
    samples = imdb_dataset.shuffle().select(range(num_samples))

    print("\nRunning inference on selected samples:\n")
    for i, sample in enumerate(samples):
        text = sample["text"]
        true_label_id = sample["label"]
        true_label = "positive" if true_label_id == 1 else "negative"
        
        print(f"--- Sample {i+1}/{num_samples} ---")
        print(f"Text: {text[:200]}...") # Print first 200 chars for brevity
        print(f"True Sentiment: {true_label}")
        
        prediction = inferer.predict(text)
        print(f"Predicted Sentiment: {prediction['sentiment']}")
        print(f"Confidence: {prediction['confidence']:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sample inference on IMDB dataset.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (e.g., config.yaml)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples from IMDB test set to run inference on."
    )
    args = parser.parse_args()
    run_sample_inference(config_path=args.config_path, num_samples=args.num_samples) 