from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer
import jax
import jax.numpy as jnp
import haiku as hk
from model_architecture import VishwamAIModel
import logging

app = Flask(__name__)

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token

# Lazy model initialization
def initialize_model():
    try:
        def model_fn(inputs):
            model = VishwamAIModel()
            return model(inputs)

        transformed_model_fn = hk.transform(model_fn)
        rng = jax.random.PRNGKey(42)
        example_input = jnp.array([[0]])  # Dummy input for initialization
        params = transformed_model_fn.init(rng, example_input)
        app.logger.debug("Model initialized successfully.")
        return transformed_model_fn, params, rng
    except Exception as e:
        app.logger.error(f"Error during model initialization: {e}")
        raise

# Initialize the model once when the server starts
transformed_model_fn, params, rng = initialize_model()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('input')
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Tokenize the input
        tokenized_input = tokenizer(user_input, return_tensors="jax", padding=True, truncation=True).input_ids
        tokenized_input = jax.numpy.array(tokenized_input, dtype=jnp.int32)  # Ensure inputs are integer dtype for embedding layer

        # Generate response
        output = transformed_model_fn.apply(params, rng, tokenized_input)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Error during request handling: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.logger.setLevel(logging.DEBUG)
    app.run(host='0.0.0.0', port=5000)
