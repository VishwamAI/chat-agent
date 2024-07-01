import re
from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_bias(text):
    """
    Analyze the given text for potential biases using sentiment analysis and keyword detection.

    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary containing the analysis results, including sentiment and detected biases.
    """
    # Perform sentiment analysis
    sentiment = sentiment_analyzer(text)

    # Define keywords and phrases that may indicate bias
    bias_keywords = [
        "stereotype", "prejudice", "discrimination", "biased", "unfair", "inequality",
        "racism", "sexism", "ageism", "homophobia", "transphobia", "xenophobia"
    ]

    # Detect bias keywords in the text
    detected_biases = [keyword for keyword in bias_keywords if re.search(rf"\b{keyword}\b", text, re.IGNORECASE)]

    # Compile analysis results
    analysis_results = {
        "sentiment": sentiment,
        "detected_biases": detected_biases
    }

    return analysis_results

def main():
    # Example text for analysis
    example_text = "This is an example text that may contain biased language or sentiments."

    # Analyze the example text for biases
    results = analyze_bias(example_text)

    # Print the analysis results
    print("Bias Analysis Results:")
    print(f"Sentiment: {results['sentiment']}")
    print(f"Detected Biases: {', '.join(results['detected_biases']) if results['detected_biases'] else 'None'}")

if __name__ == "__main__":
    main()
