import logging
from query_processing import process_query
from content_generation import generate_content, evaluate_relevance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_search_engine(query, max_length=100, relevance_threshold=0.7):
    """
    Simulates a user query and captures the search engine's response.
    """
    logger.info(f"Testing search engine with query: {query}")

    # Process the query
    processed_query = process_query(query)

    # Generate content based on the processed query
    response = generate_content(processed_query, max_length=max_length)

    # Log the response
    if "generated_content" in response:
        generated_content = response["generated_content"]
        logger.info(f"Generated content: {generated_content}")

        # Evaluate relevance of the generated content
        is_relevant = evaluate_relevance(query, generated_content, threshold=relevance_threshold)
        logger.info(f"Relevance: {'Relevant' if is_relevant else 'Not Relevant'}")
    else:
        logger.error(f"Error: {response['error']}")

    return response

if __name__ == "__main__":
    # Example queries for testing
    queries = [
        "What is the impact of generative AI on search engines?",
        "How does GPT-3 differ from GPT-2?",
        "What are the latest advancements in image generation?",
        "Can generative AI create music?",
        "What are the ethical considerations of using generative AI?",
        "How can generative AI be used in healthcare?",
        "What are the challenges of using generative AI in finance?",
        "Can generative AI improve customer service?",
        "What are the environmental impacts of generative AI?",
        "How does generative AI affect job markets?"
    ]

    # Run the test for each query
    for query in queries:
        response = test_search_engine(query)
        print(f"Query: {query}\nResponse: {response}\n")
