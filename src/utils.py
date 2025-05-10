import datetime
import os
from pathlib import Path
import re # Added for parsing

def get_loss_acronym(loss_function_name: str) -> str:
    """Generates an acronym for the loss function."""
    if "SentimentFocalLoss".lower() in loss_function_name.lower():
        return "SFL"
    elif "SentimentWeightedLoss".lower() in loss_function_name.lower():
        return "SWL"
    # Fallback for unknown loss functions - can be expanded
    name_parts = [part[0] for part in loss_function_name.split() if part]
    if name_parts:
        return "".join(name_parts).upper()
    return "LOSSFUNC"

def generate_artifact_name(
    base_output_dir: str | Path,
    model_config_name: str, # This should be the simple name, e.g., ModernBERT-base
    loss_function_name: str, # Full name like "SentimentFocalLoss"
    epoch: int,
    artifact_type: str, 
    timestamp_str: str | None = None,
    f1_score: float | None = None,
    plot_description: str | None = None, 
    extension: str | None = None 
) -> Path:
    """
    Generates a consistent artifact name based on the defined convention.
    Prefix: {model_name}_{timestamp}_{loss_acronym}_e{epoch#}_
    F1 score ({f1score}f1_) is included for "checkpoint" and "plot_confusion_matrix".
    model_config_name should be the base name (e.g., ModernBERT-base, not the full HF path).
    """
    if timestamp_str is None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Ensure simple_model_name is used, not the full HF path if passed by mistake.
    simple_model_name = model_config_name.split('/')[-1]

    loss_acronym = get_loss_acronym(loss_function_name)

    base_filename_parts = [
        simple_model_name,
        timestamp_str,
        loss_acronym,
        f"e{epoch}",
    ]

    if artifact_type in ["checkpoint", "plot_confusion_matrix"]:
        if f1_score is not None:
            base_filename_parts.append(f"{f1_score:.4f}f1")
        else:
            # Forcing F1 to be present for these types. Caller should ensure it.
            # If it's absolutely not available, it might indicate an issue upstream.
            # Adding a placeholder like NOF1 can hide problems.
            # Consider raising an error or handling it explicitly in the caller.
            # For now, let's stick to the original placeholder for consistency if tests expect it.
            base_filename_parts.append("NOF1") 

    suffix_parts = []
    if artifact_type == "checkpoint":
        suffix_parts.append("checkpoint")
    elif artifact_type == "metrics":
        suffix_parts.append("metrics")
    elif artifact_type == "plot_confusion_matrix":
        suffix_parts.append("confusion_matrix")
    elif artifact_type.startswith("plot_"):
        description = plot_description if plot_description else artifact_type.replace("plot_", "")
        suffix_parts.append(description)
    else:
        suffix_parts.append(artifact_type)
    
    # Join base parts first, then add the F1 part if it exists (it's already in base_filename_parts if applicable)
    # Then join with suffix parts.
    filename_core = "_".join(base_filename_parts)
    final_filename_str = f"{filename_core}_{'_'.join(suffix_parts)}" if suffix_parts else filename_core

    if extension:
        filename_with_ext = f"{final_filename_str}.{extension.lstrip('.')}"
    else:
        if artifact_type == "checkpoint":
            filename_with_ext = f"{final_filename_str}.pt"
        elif artifact_type == "metrics":
            filename_with_ext = f"{final_filename_str}.json"
        elif artifact_type.startswith("plot_"):
            filename_with_ext = f"{final_filename_str}.png"
        else:
            filename_with_ext = final_filename_str

    return Path(base_output_dir) / filename_with_ext


# Regex to parse the new artifact filenames
# Pattern: {model_name}_{timestamp}_{loss_acronym}_e{epoch_num}[_{f1_score}f1]_{artifact_suffix}.{ext}
# Example: ModernBERT-base_20230101000000_SFL_e50_0.1234f1_checkpoint.pt
# Example: ModernBERT-base_20230101000000_SWL_e75_metrics.json
# Example: ModernBERT-base_20230101000000_SFL_e50_loss_curve.png
FILENAME_PARSE_REGEX = re.compile(
    r"^(?P<model_name>[^_]+)_"
    r"(?P<timestamp>\d{14})_"
    r"(?P<loss_acronym>[A-Z0-9]+)_"
    r"e(?P<epoch>\d+)"
    r"(?:_(?P<f1_score>\d+\.\d{4})f1)?_?"
    r"(?P<artifact_suffix>.+?)"
    r"\.(?P<extension>[a-zA-Z0-9]+)$"
)

def parse_artifact_filename(filename: str) -> dict | None:
    """Parses an artifact filename string to extract its components.

    Args:
        filename: The filename string (e.g., 'ModelName_20230101000000_SFL_e10_0.1234f1_checkpoint.pt')

    Returns:
        A dictionary with keys: 'model_name', 'timestamp', 'loss_acronym', 'epoch',
        'f1_score' (optional, float), 'artifact_suffix', 'extension'.
        Returns None if the filename doesn_match the expected pattern.
    """
    match = FILENAME_PARSE_REGEX.match(filename)
    if not match:
        return None
    
    parts = match.groupdict()
    parts['epoch'] = int(parts['epoch'])
    if parts['f1_score']:
        parts['f1_score'] = float(parts['f1_score'])
    else:
        parts['f1_score'] = None # Ensure it's None if not present
        
    return parts


if __name__ == '__main__':
    # Example Usage (for testing this script directly)
    output_dir = "test_artifacts"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Corrected model_name for generate_artifact_name calls
    hf_model_name_bert = "answerdotai/ModernBERT-base"
    simple_model_name_bert = hf_model_name_bert.split('/')[-1]
    hf_model_name_deberta = "microsoft/deberta-v3-small"
    simple_model_name_deberta = hf_model_name_deberta.split('/')[-1]
    
    print("--- Testing generate_artifact_name ---")

    cp_name = generate_artifact_name(
        base_output_dir=output_dir,
        model_config_name=simple_model_name_bert, # Use simple name
        loss_function_name="SentimentFocalLoss",
        epoch=50,
        artifact_type="checkpoint",
        f1_score=0.875123,
        extension="pt"
    )
    print(f"Generated Checkpoint: {cp_name}")

    metrics_name = generate_artifact_name(
        base_output_dir=output_dir,
        model_config_name=simple_model_name_bert, # Use simple name
        loss_function_name="SentimentWeightedLoss",
        epoch=75,
        artifact_type="metrics",
        extension="json"
    )
    print(f"Generated Metrics JSON: {metrics_name}")

    cm_plot_name = generate_artifact_name(
        base_output_dir=output_dir,
        model_config_name=simple_model_name_deberta, # Use simple name
        loss_function_name="SentimentFocalLoss",
        epoch=50,
        artifact_type="plot_confusion_matrix",
        f1_score=0.9218,
        extension="png"
    )
    print(f"Generated Confusion Matrix Plot: {cm_plot_name}")

    loss_curve_name = generate_artifact_name(
        base_output_dir=output_dir,
        model_config_name=simple_model_name_bert, # Use simple name
        loss_function_name="SentimentWeightedLoss",
        epoch=75,
        artifact_type="plot_loss_curve",
        extension="svg"
    )
    print(f"Generated Loss Curve Plot: {loss_curve_name}")

    print("\n--- Testing parse_artifact_filename ---")
    test_names = [
        str(cp_name.name),
        str(metrics_name.name),
        str(cm_plot_name.name),
        str(loss_curve_name.name),
        "OldNameFormat_epoch10.pt", # Should fail
        "MyModel_20240101120000_SFL_e5_custom_report.txt", # Generic suffix
        "MyModel_20240101120000_SFL_e5_0.9999f1_checkpoint.bin"
    ]
    for name_str in test_names:
        parsed = parse_artifact_filename(name_str)
        if parsed:
            print(f"Parsed '{name_str}': {parsed}")
        else:
            print(f"Failed to parse '{name_str}'")

    # Test parsing of a name generated by the updated function logic
    # Corrected join logic test
    complex_plot_name = generate_artifact_name(
        base_output_dir=output_dir,
        model_config_name=simple_model_name_bert,
        loss_function_name="SentimentFocalLoss",
        epoch=10,
        artifact_type="plot_custom_metric", # Will become suffix part
        plot_description="my_special_metric_over_time", # Takes precedence for plot_
        extension="csv"
    )
    print(f"Generated Custom Plot: {complex_plot_name}")
    parsed_complex = parse_artifact_filename(complex_plot_name.name)
    print(f"Parsed Custom Plot '{complex_plot_name.name}': {parsed_complex}")

    # Test case for filename where f1 is not applicable, e.g. metrics.json
    # The regex for f1 (?:_(?P<f1_score>\d+\.\d{4})f1)? handles its optionality.
    # The suffix part of the regex _?(?P<artifact_suffix>.+?) might need adjustment if the underscore is not always there before suffix
    # Current generate_artifact_name adds it: f"{filename_core}_{'_'.join(suffix_parts)}"
    # Let's test a metrics filename string directly:
    test_metrics_str = "ModernBERT-base_20240523100000_SWL_e75_metrics.json"
    parsed_test_metrics = parse_artifact_filename(test_metrics_str)
    print(f"Parsed Test Metrics Str '{test_metrics_str}': {parsed_test_metrics}")
    assert parsed_test_metrics is not None
    assert parsed_test_metrics['f1_score'] is None

    # Test a checkpoint name with f1
    test_ckpt_str = "ModernBERT-base_20240523110000_SFL_e50_0.8888f1_checkpoint.pt"
    parsed_test_ckpt = parse_artifact_filename(test_ckpt_str)
    print(f"Parsed Test Ckpt Str '{test_ckpt_str}': {parsed_test_ckpt}")
    assert parsed_test_ckpt is not None
    assert parsed_test_ckpt['f1_score'] == 0.8888

    print("--- Test complete ---")
    # Create dummy files based on generated names to check
    # Path(cp_name).touch()
    # Path(metrics_name).touch()
    # Path(cm_plot_name).touch()
    # Path(loss_curve_name).touch()
    # Path(complex_plot_name).touch()
    # Path(test_metrics_str).touch()
    # Path(test_ckpt_str).touch() 