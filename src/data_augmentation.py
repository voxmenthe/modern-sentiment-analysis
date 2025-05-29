import math

def _truncate_prefix(text: str, target_char_len: int) -> str:
    """Cuts off characters from the beginning of the text."""
    if len(text) <= target_char_len:
        return text
    return text[len(text) - target_char_len:]

def _truncate_suffix(text: str, target_char_len: int) -> str:
    """Cuts off characters from the end of the text."""
    if len(text) <= target_char_len:
        return text
    return text[:target_char_len]

def _truncate_both_ends(text: str, target_char_len: int) -> str:
    """Cuts off characters equally from both ends of the text."""
    if len(text) <= target_char_len:
        return text
    
    chars_to_remove = len(text) - target_char_len
    remove_from_start = chars_to_remove // 2
    remove_from_end = chars_to_remove - remove_from_start
    
    start_index = remove_from_start
    end_index = len(text) - remove_from_end
    
    return text[start_index:end_index]

def apply_char_truncation_and_adjust_label(
    text: str, 
    label: int, 
    char_truncation_percentage: float, 
    truncation_method: str
) -> tuple[str, float]:
    """
    Applies character-level truncation to the text and adjusts the label accordingly.

    Args:
        text: The original text string.
        label: The original label (0 or 1).
        char_truncation_percentage: The percentage of the original length to truncate to (e.g., 0.8 for 80%).
        truncation_method: The method to use for truncation ("prefix", "suffix", or "both_ends").

    Returns:
        A tuple containing the truncated text and the adjusted label.
    """
    original_char_len = len(text)
    if original_char_len == 0: # Avoid division by zero for empty strings
        return text, float(label)

    target_char_len = math.ceil(original_char_len * char_truncation_percentage)

    if target_char_len >= original_char_len: # No truncation needed if percentage results in same or greater length
        return text, float(label)

    truncated_text: str
    if truncation_method == "prefix":
        truncated_text = _truncate_prefix(text, target_char_len)
    elif truncation_method == "suffix":
        truncated_text = _truncate_suffix(text, target_char_len)
    elif truncation_method == "both_ends":
        truncated_text = _truncate_both_ends(text, target_char_len)
    else:
        raise ValueError(f"Unknown truncation_method: {truncation_method}")

    if len(text) == 0: # Should not happen if original_char_len > 0 and target_char_len < original_char_len
        length_ratio = 0.0
    else:
        length_ratio = len(truncated_text) / original_char_len
    
    adjusted_label: float
    if label == 1:
        adjusted_label = 1.0 * length_ratio
    elif label == 0:
        adjusted_label = 0.0 + (1.0 - length_ratio)
    else:
        # This case should ideally not happen with binary labels, but good to be robust
        adjusted_label = float(label) 

    return truncated_text, adjusted_label 