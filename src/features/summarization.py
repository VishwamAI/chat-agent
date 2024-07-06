from transformers import pipeline

def summarize_text(text: str, max_length: int = 150, min_length: int = 30) -> str:
    """
    Summarizes the given text using a pre-trained summarization model from Hugging Face.

    Args:
        text (str): The text to be summarized.
        max_length (int): The maximum length of the summary.
        min_length (int): The minimum length of the summary.

    Returns:
        str: The summarized text.
    """
    try:
        summarizer = pipeline("summarization")
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {e}"
