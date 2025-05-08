import gradio as gr
from inference import SentimentInference
import os
from datasets import load_dataset
import random
import torch
from torch.utils.data import DataLoader
from evaluation import evaluate
from tqdm import tqdm

# --- Initialize Sentiment Model ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = "config.yaml"
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Configuration file not found. Tried {os.path.join(os.path.dirname(__file__), 'config.yaml')} and {CONFIG_PATH}. "
            f"Ensure 'config.yaml' exists and is accessible."
        )

print(f"Loading model with config: {CONFIG_PATH}")
try:
    sentiment_inferer = SentimentInference(config_path=CONFIG_PATH)
    print("Sentiment model loaded successfully.")
except Exception as e:
    print(f"Error loading sentiment model: {e}")
    sentiment_inferer = None

# --- Load IMDB Dataset ---
print("Loading IMDB dataset for samples...")
try:
    imdb_dataset = load_dataset("imdb", split="test")
    print("IMDB dataset loaded successfully.")
except Exception as e:
    print(f"Failed to load IMDB dataset: {e}. Sample loading will be disabled.")
    imdb_dataset = None

def load_random_imdb_sample():
    """Loads a random sample text from the IMDB dataset."""
    if imdb_dataset is None:
        return "IMDB dataset not available. Cannot load sample.", None
    random_index = random.randint(0, len(imdb_dataset) - 1)
    sample = imdb_dataset[random_index]
    return sample["text"], sample["label"]

def predict_sentiment(text_input, true_label_state):
    """Predicts sentiment for the given text_input."""
    if sentiment_inferer is None:
        return "Error: Sentiment model could not be loaded. Please check the logs.", true_label_state
    
    if not text_input or not text_input.strip():
        return "Please enter some text for analysis.", true_label_state
    
    try:
        prediction = sentiment_inferer.predict(text_input)
        sentiment = prediction['sentiment']
        
        # Convert numerical label to text if available
        true_sentiment = None
        if true_label_state is not None:
            true_sentiment = "positive" if true_label_state == 1 else "negative"
        
        result = f"Predicted Sentiment: {sentiment.capitalize()}"
        if true_sentiment:
            result += f"\nTrue IMDB Label: {true_sentiment.capitalize()}"
        
        return result, None  # Reset true label state after display
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error during prediction: {str(e)}", true_label_state

def run_full_evaluation_gradio():
    """Runs full evaluation on the IMDB test set and yields results for Gradio."""
    if sentiment_inferer is None or sentiment_inferer.model is None:
        yield "Error: Sentiment model could not be loaded. Cannot run evaluation."
        return

    try:
        accumulated_text = "Starting full evaluation... This will process 25,000 samples and may take 10-20 minutes. Please be patient.\n"
        yield accumulated_text
        
        device = sentiment_inferer.device
        model = sentiment_inferer.model
        tokenizer = sentiment_inferer.tokenizer
        max_length = sentiment_inferer.max_length
        batch_size = 16  # Consistent with evaluation.py default

        yield "Loading IMDB test dataset (this might take a moment)..."
        imdb_test_full = load_dataset("imdb", split="test")
        accumulated_text += f"IMDB test dataset loaded ({len(imdb_test_full)} samples). Tokenizing dataset...\n"
        yield accumulated_text

        def tokenize_function(examples):
            tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
            tokenized_output["lengths"] = [sum(mask) for mask in tokenized_output["attention_mask"]]
            return tokenized_output
        
        tokenized_imdb_test_full = imdb_test_full.map(tokenize_function, batched=True, num_proc=os.cpu_count()//2 if os.cpu_count() > 1 else 1)
        tokenized_imdb_test_full = tokenized_imdb_test_full.remove_columns(["text"])
        tokenized_imdb_test_full = tokenized_imdb_test_full.rename_column("label", "labels")
        tokenized_imdb_test_full.set_format("torch", columns=["input_ids", "attention_mask", "labels", "lengths"])

        test_dataloader_full = DataLoader(tokenized_imdb_test_full, batch_size=batch_size)
        accumulated_text += "Dataset tokenized and DataLoader prepared. Starting model evaluation on the test set...\n"
        yield accumulated_text

        # The 'evaluate' function from evaluation.py is now a generator.
        # Iterate through its yielded updates and results, accumulating text.
        for update in evaluate(model, test_dataloader_full, device):
            if isinstance(update, dict):
                # This is the final results dictionary
                results_str = "\n--- Full Evaluation Results ---\n" # Start with a newline
                for key, value in update.items():
                    if isinstance(value, float):
                        results_str += f"{key.capitalize()}: {value:.4f}\n"
                    else:
                        results_str += f"{key.capitalize()}: {value}\n"
                results_str += "\nEvaluation finished."
                accumulated_text += results_str
                yield accumulated_text 
                break # Stop after getting the results dict
            else:
                # This is a progress string
                accumulated_text += str(update) + "\n" # Append newline to each progress string
                yield accumulated_text

    except Exception as e:
        import traceback
        error_msg = f"An error occurred during full evaluation:\n{str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        yield error_msg

# --- Gradio Interface ---
with gr.Blocks() as demo:
    true_label = gr.State()
    
    gr.Markdown("## IMDb Sentiment Analyzer")
    gr.Markdown("Enter a movie review to classify its sentiment as Positive or Negative, or load a random sample from the IMDb dataset.")
    
    with gr.Row():
        input_textbox = gr.Textbox(lines=7, placeholder="Enter movie review here...", label="Movie Review", scale=3)
        output_text = gr.Text(label="Analysis Result", scale=1)

    with gr.Row():
        submit_button = gr.Button("Analyze Sentiment")
        load_sample_button = gr.Button("Load Random IMDB Sample")

    gr.Examples(
        examples=[
            ["This movie was absolutely fantastic! The acting was superb and the plot was gripping."],
            ["I was really disappointed with this film. It was boring and the story made no sense."],
            ["An average movie, had some good parts but overall quite forgettable."],
            ["While the plot was predictable, the acting was solid and the plot was engaging. Overall it was watchable"]
        ],
        inputs=input_textbox
    )

    with gr.Accordion("Advanced: Full Model Evaluation on IMDB Test Set", open=False):
        gr.Markdown(
            """**WARNING!** Clicking the button below will run the sentiment analysis model on the **entire IMDB test dataset (25,000 reviews)**. "
            
            "This is computationally intensive process and will take a long time (potentially **20 minutes or more** depending on the hardware of the Hugging Face Space or machine running this app). It may not even run unless the hardware is upgraded. "
            
            "The application might appear unresponsive during this period. "
            
            "Progress messages will be shown below."""
        )
        run_eval_button = gr.Button("Run Full Evaluation on IMDB Test Set")
        evaluation_output_textbox = gr.Textbox(
            label="Evaluation Progress & Results",
            lines=15,
            interactive=False,
            show_label=True,
            max_lines=20
        )
        run_eval_button.click(
            fn=run_full_evaluation_gradio, 
            inputs=None, 
            outputs=evaluation_output_textbox
        )

    # Wire actions
    submit_button.click(
        fn=predict_sentiment,
        inputs=[input_textbox, true_label],
        outputs=[output_text, true_label]
    )
    load_sample_button.click(
        fn=load_random_imdb_sample,
        inputs=None,
        outputs=[input_textbox, true_label]
    )

if __name__ == '__main__':
    print("Launching Gradio interface...")
    demo.launch(share=False)
