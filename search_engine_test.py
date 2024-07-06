import logging
from query_processing import process_query
from content_generation import generate_content

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_search_engine(query, max_length=100):
    """
    Simulates a user query and captures the search engine's response.
    """
    logger.info(f"Testing search engine with query: {query}")

    # Process the query
    processed_query = process_query(query)

    # Generate content based on the processed query
    response = generate_content(processed_query, max_length=max_length)

    # Log the response
    logger.info(f"Search engine response: {response}")

    return response

if __name__ == "__main__":
    # Example queries for testing
    queries = [
        "What is the impact of generative AI on search engines?",
        "How does GPT-3 differ from GPT-2?",
        "What are the latest advancements in image generation?",
        "Can generative AI create music?",
        "What are the ethical considerations of using generative AI?"
    ]

    # Run the test for each query
    for query in queries:
        response = test_search_engine(query)
        print(f"Query: {query}\nResponse: {response}\n")
