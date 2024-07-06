from transformers import pipeline, set_seed
import logging
import random
import spacy

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load the pre-trained GPT model
try:
    generator = pipeline('text-generation', model='gpt2')
except Exception as e:
    logger.error(f"Error loading GPT model: {e}")
    generator = None

# Load the spaCy model for keyword extraction
try:
    nlp = spacy.load("en_core_web_lg")
except spacy.errors.ModelNotFoundError as e:
    logger.error(f"Model not found: {e}")
    nlp = None
except Exception as e:
    logger.error(f"Error loading spaCy model: {e}")
    nlp = None

def generate_content(processed_query: dict, max_length: int = 100) -> dict:
    """
    Generate content based on the processed search query using a pre-trained GPT model.

    Args:
    processed_query (dict): The processed search query containing tokens, pos_tags, and named_entities.
    max_length (int): The maximum length of the generated content. Must be a positive integer and less than or equal to 500.

    Returns:
    dict: A dictionary containing the generated content or an error message.
        - "generated_content": The generated content based on the query.
        - "error": An error message if content generation fails or if input is invalid.
    """
    if generator is None:
        logger.error("GPT model not loaded")
        return {"error": "GPT model not loaded"}

    if "tokens" not in processed_query:
        logger.error("Processed query does not contain 'tokens' key")
        return {"error": "Processed query does not contain 'tokens' key"}

    if not isinstance(max_length, int) or max_length <= 0 or max_length > 500:
        logger.error("Invalid max_length parameter. It must be a positive integer and less than or equal to 500.")
        return {"error": "Invalid max_length parameter. It must be a positive integer and less than or equal to 500."}

    try:
        # Extract the query text from the processed query
        query_text = " ".join(processed_query["tokens"])
        logger.info(f"Query text: {query_text}")

        # Use a random seed for content generation to ensure diverse outputs
        random_seed = random.randint(0, 10000)
        set_seed(random_seed)
        logger.info(f"Random seed: {random_seed}")

        # Generate content using the GPT model
        generated = generator(query_text, max_length=max_length, num_return_sequences=1, truncation=True)
        generated_text = generated[0]['generated_text']
        logger.info(f"Generated text: {generated_text}")

        # Check the quality and length of the generated content
        if len(generated_text) < 20:
            logger.error("Generated content is too short")
            return {"error": "Generated content is too short"}
        if "error" in generated_text.lower():
            logger.error("Generated content contains errors")
            return {"error": "Generated content contains errors"}

        # Check for relevance of the generated content to the input query
        if not evaluate_relevance(query_text, generated_text):
            logger.error("Generated content is not relevant to the input query")
            return {"error": "Generated content is not relevant to the input query"}

        return {"generated_content": generated_text}
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return {"error": "An error occurred during content generation"}


def evaluate_relevance(query: str, generated_content: str, threshold: float = 0.7) -> bool:
    """
    Evaluate the relevance of the generated content to the original query.

    Args:
    query (str): The original user query.
    generated_content (str): The generated content.
    threshold (float): The relevance threshold based on semantic similarity. Default is 0.7.

    Returns:
    bool: True if the generated content is relevant to the query, False otherwise.
    """
    if nlp is None:
        logger.error("spaCy model not loaded")
        return False

    query_doc = nlp(query)
    content_doc = nlp(generated_content)
    similarity = query_doc.similarity(content_doc)
    logger.info(f"Semantic similarity: {similarity}")
    return similarity >= threshold
