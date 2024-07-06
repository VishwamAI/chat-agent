# Generative AI Search Engine

This project is a generative AI search engine that uses advanced natural language processing (NLP) techniques to generate relevant content based on user queries. The search engine leverages pre-trained models from the `transformers` library and spaCy for keyword extraction and relevance assessment.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/VishwamAI/chat-agent.git
   cd generative_ai_project
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the spaCy English language model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

### Running the Search Engine

1. Ensure that the virtual environment is activated:

   ```bash
   source venv/bin/activate
   ```

2. Run the search engine test script:

   ```bash
   python search_engine_test.py
   ```

### Project Structure

- `content_generation.py`: Contains the `generate_content` function for generating content based on processed queries.
- `query_processing.py`: Contains the `process_query` function for processing user queries.
- `search_engine_test.py`: Script for testing the search engine with example queries.
- `requirements.txt`: Lists the required dependencies for the project.

### Dependencies

- `tensorflow==2.16.2`
- `spacy==3.7.5`
- `en-core-web-sm==3.7.1`
- `torch==2.0.1`
- `transformers==4.30.2`

### Notes

- The `generate_content` function uses a pre-trained GPT-2 model from the `transformers` library.
- The `extract_keywords` function uses spaCy for keyword extraction, including lemmatization and stop word removal.
- The `evaluate_relevance` function assesses the relevance of generated content based on a configurable threshold.

## Next Steps

- Refine the `evaluate_relevance` function to improve the accuracy of relevance assessment.
- Adjust the generation parameters to better align the generated content with the input queries.
- Implement a configurable relevance threshold to allow fine-tuning of relevance criteria.
- Enhance the `extract_keywords` function with more sophisticated NLP techniques for better keyword extraction.
- Add comprehensive error handling and logging to improve debugging and maintainability.

## Contact

For any questions or feedback, please contact the project maintainer.
