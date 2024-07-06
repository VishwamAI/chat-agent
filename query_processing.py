import spacy
import logging
from typing import Dict, Union, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except spacy.errors.ModelNotFoundError as e:
    logger.error(f"Model not found: {e}")
    nlp = None
except Exception as e:
    logger.error(f"Error loading spaCy model: {e}")
    nlp = None

def process_query(query: str) -> Dict[str, Union[str, List[str], List[Tuple[str, str]], Dict[str, str]]]:
    """
    Process the user input query to extract relevant information and intent.

    Args:
    query (str): The user input query.

    Returns:
    dict: A dictionary containing the processed query information or an error message.
    """
    if nlp is None:
        return {"error": "spaCy model not loaded"}

    if not isinstance(query, str) or not query.strip():
        return {"error": "Invalid query. Please provide a non-empty string."}

    try:
        doc = nlp(query)

        # Extract tokens, part-of-speech tags, and named entities
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]

        processed_query = {
            "tokens": tokens,
            "pos_tags": pos_tags,
            "named_entities": named_entities
        }

        return processed_query
    except spacy.errors.LanguageError as e:
        return {"error": f"Language processing error: {e}"}
    except Exception as e:
        return {"error": f"Error processing query: {e}"}

if __name__ == "__main__":
    # Example usage
    query = "What is the impact of generative AI on search engines?"
    result = process_query(query)
    if "error" in result:
        logger.error(f"Error: {result['error']}")
    else:
        logger.info(result)
