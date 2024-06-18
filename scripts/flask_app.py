from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer
import jax
import jax.numpy as jnp
import haiku as hk
from model_architecture import VishwamAIModel
import logging
import sys
import os

app = Flask(__name__)

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token

# Lazy model initialization
transformed_model_fn = None
params = None
rng = None

# State management for conversation context
conversation_context = {}

def initialize_model():
    global transformed_model_fn, params, rng
    try:
        app.logger.debug(f"sys.path: {sys.path}")  # Log the sys.path for debugging
        app.logger.debug(f"Environment PATH: {os.environ['PATH']}")  # Log the PATH environment variable for debugging
        app.logger.debug(f"Environment PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")  # Log the PYTHONPATH environment variable for debugging
        app.logger.debug("Initializing model...")  # Unique log message for confirmation
        def model_fn(inputs):
            model = VishwamAIModel()
            return model(inputs)

        transformed_model_fn = hk.transform(model_fn)
        rng = jax.random.PRNGKey(42)
        example_input = jnp.array([[0]])  # Dummy input for initialization
        params = transformed_model_fn.init(rng, example_input)
        app.logger.debug("Model initialized successfully.")
    except Exception as e:
        app.logger.error(f"Error during model initialization: {e}")
        raise

@app.route('/chat', methods=['POST'])
def chat():
    global transformed_model_fn, params, rng, conversation_context
    try:
        if transformed_model_fn is None or params is None or rng is None:
            initialize_model()

        user_id = request.json.get('user_id')
        user_input = request.json.get('input')
        if not user_input or not user_id:
            return jsonify({"error": "No input or user_id provided"}), 400

        # Maintain conversation context
        if user_id not in conversation_context:
            conversation_context[user_id] = []

        conversation_context[user_id].append(user_input)

        # Tokenize the input
        tokenized_input = tokenizer(user_input, return_tensors="jax", padding=True, truncation=True).input_ids
        tokenized_input = jax.numpy.array(tokenized_input, dtype=jnp.int32)  # Ensure inputs are integer dtype for embedding layer

        # Generate response
        output = transformed_model_fn.apply(params, rng, tokenized_input)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Update conversation context with the response
        conversation_context[user_id].append(response)

        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Error during request handling: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.logger.setLevel(logging.DEBUG)
    print(f"sys.path at startup: {sys.path}")  # Print sys.path at startup for debugging
    app.run(host='0.0.0.0', port=5000)
