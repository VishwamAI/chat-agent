from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer
import jax
import jax.numpy as jnp
import haiku as hk
from scripts.model_architecture import VishwamAIModel
import logging
import os
import numpy as np
from scripts.vishwamai_prototype import VishwamAI

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

@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        input_text = request.json.get('input_text')
        resolution = request.json.get('resolution', [512, 512])
        if not input_text:
            return jsonify({"error": "No input_text provided"}), 400

        app.logger.debug(f"Received request to generate image with input_text: {input_text} and resolution: {resolution}")

        vishwamai = VishwamAI(batch_size=16)
        app.logger.debug(f"VishwamAI object created: {vishwamai}")
        app.logger.debug(f"VishwamAI methods: {dir(vishwamai)}")

        generated_image = vishwamai.generate_image(input_text, target_resolution=resolution)
        if generated_image is None:
            app.logger.error("Failed to generate image: generated_image is None")
            return jsonify({"error": "Failed to generate image"}), 500

        app.logger.debug("Image generated successfully")

        # Denormalize the image to [0, 255] range
        generated_image = (generated_image * 127.5 + 127.5).astype(np.uint8)

        # Convert the image to a list for JSON serialization
        image_list = generated_image.tolist()

        app.logger.debug("Image converted to list for JSON serialization")

        return jsonify({"generated_image": image_list})
    except Exception as e:
        app.logger.error(f"Error during image generation: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Configure logging to write to a file
    logging.basicConfig(level=logging.DEBUG)
    file_handler = logging.FileHandler('logs/server.log')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)

    app.logger.setLevel(logging.DEBUG)
    app.run(host='0.0.0.0', port=5000)
