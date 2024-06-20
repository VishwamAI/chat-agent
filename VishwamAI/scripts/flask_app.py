from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer
import jax
import jax.numpy as jnp
import haiku as hk
from .model_architecture import VishwamAIModel
import logging
import sys
import os
import subprocess
import traceback  # Add this import statement
from datetime import datetime  # Add this import statement

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# Add file handler to write logs to a file
file_handler = logging.FileHandler('flask_app.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)

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
    app.logger.debug(f"JAX import check: {jax}")  # Confirm the import of JAX
    try:
        app.logger.debug("Starting model initialization...")  # Unique log message for confirmation
        app.logger.debug(f"Python interpreter: {sys.executable}")  # Log the Python interpreter for debugging
        app.logger.debug(f"Python interpreter path: {subprocess.check_output(['which', 'python3']).decode('utf-8').strip()}")  # Log the path of the Python interpreter for debugging
        app.logger.debug(f"Python version: {subprocess.check_output(['python3', '--version']).decode('utf-8').strip()}")  # Log the Python version for debugging
        app.logger.debug(f"sys.path: {sys.path}")  # Log the sys.path for debugging
        app.logger.debug(f"Environment PATH: {os.environ['PATH']}")  # Log the PATH environment variable for debugging
        app.logger.debug(f"Environment PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")  # Log the PYTHONPATH environment variable for debugging
        installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'list']).decode('utf-8')
        app.logger.debug(f"Installed packages before initialization: {installed_packages}")  # Log the installed packages before initialization
        app.logger.debug("Initializing model...")  # Unique log message for confirmation

        def model_fn(inputs):
            model = VishwamAIModel()
            return model(inputs)

        transformed_model_fn = hk.transform(model_fn)
        rng = jax.random.PRNGKey(42)
        example_input = ["This is a dummy input for model initialization."]  # Dummy input for initialization as a list of strings
        app.logger.debug(f"Example input: {example_input}")  # Log the example input for debugging
        try:
            app.logger.debug(f"Initializing model with input: {example_input}")
            app.logger.debug(f"Type of example input: {type(example_input)}")  # Log the type of example input for debugging
            app.logger.debug(f"Content of example input: {example_input}")  # Log the content of example input for debugging
            if not isinstance(example_input, list) or not all(isinstance(i, str) for i in example_input):
                raise ValueError("Example input must be a list of strings")
            tokenized_input = tokenizer(example_input, padding=True, truncation=True).input_ids
            app.logger.debug(f"Tokenized input before JAX conversion: {tokenized_input}")  # Log the tokenized input before JAX conversion
            tokenized_input = [list(map(int, token)) for token in tokenized_input]  # Convert tokenized input to List[List[int]]
            app.logger.debug(f"Tokenized input after JAX conversion: {tokenized_input}")  # Log the tokenized input after JAX conversion
            app.logger.debug(f"Dtype of tokenized input: {type(tokenized_input[0][0])}")  # Log the dtype of tokenized input
            global params  # Ensure params is treated as global
            params = transformed_model_fn.init(rng, tokenized_input)
            app.logger.debug("Model initialized successfully.")
        except Exception as init_error:
            app.logger.error(f"Error during model parameter initialization: {init_error}")
            app.logger.error(f"Input during initialization error: {example_input}")
            raise
    except Exception as e:
        app.logger.error(f"Error during model initialization: {e}")
        app.logger.error(f"Stack trace: {traceback.format_exc()} - TIMESTAMP: {datetime.now().isoformat()}")
        app.logger.error(f"sys.path during error: {sys.path}")  # Log the sys.path during error for debugging
        app.logger.error(f"Environment PATH during error: {os.environ['PATH']}")  # Log the PATH environment variable during error for debugging
        app.logger.error(f"Environment PYTHONPATH during error: {os.environ.get('PYTHONPATH', '')}")  # Log the PYTHONPATH environment variable during error for debugging
        try:
            python_interpreter = sys.executable
            installed_packages = subprocess.check_output([python_interpreter, '-m', 'pip', 'list']).decode('utf-8')
            app.logger.error(f"Python interpreter: {python_interpreter}")
            app.logger.error(f"Installed packages: {installed_packages}")
        except Exception as pkg_error:
            app.logger.error(f"Error retrieving installed packages: {pkg_error} - TIMESTAMP: {datetime.now().isoformat()}")
        raise

@app.route('/chat', methods=['POST'])
def chat():
    app.logger.debug("Incoming request data: %s", request.json)  # Log the incoming request data
    app.logger.debug("Test log entry: /chat endpoint hit")  # Test log entry to confirm logging is working
    global transformed_model_fn, params, rng, conversation_context
    try:
        app.logger.debug(f"Python interpreter: {sys.executable}")  # Log the Python interpreter for debugging
        app.logger.debug(f"Python interpreter path: {subprocess.check_output(['which', 'python3']).decode('utf-8').strip()}")  # Log the path of the Python interpreter for debugging
        app.logger.debug(f"Python version: {subprocess.check_output(['python3', '--version']).decode('utf-8').strip()}")  # Log the Python version for debugging
        app.logger.debug(f"sys.path: {sys.path}")  # Log the sys.path for debugging
        app.logger.debug(f"Environment PATH: {os.environ['PATH']}")  # Log the PATH environment variable for debugging
        app.logger.debug(f"Environment PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")  # Log the PYTHONPATH environment variable for debugging
        installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'list']).decode('utf-8')
        app.logger.debug(f"Installed packages before request handling: {installed_packages}")  # Log the installed packages before request handling

        if transformed_model_fn is None or params is None or rng is None:
            with app.app_context():
                initialize_model()

        user_id = request.json.get('user_id')
        user_input = request.json.get('input')
        if not user_input or not user_id:
            return jsonify({"error": "No input or user_id provided"}), 400

        # Log the type and content of user_input before format check
        app.logger.debug(f"CHAT_ENDPOINT - Type of user_input before format check: {type(user_input)}")
        app.logger.debug(f"CHAT_ENDPOINT - Content of user_input before format check: {user_input}")

        # Ensure user_input is in the correct format
        if isinstance(user_input, str):
            user_input = [user_input]  # Convert single input to a batch of one
        elif isinstance(user_input, list) and all(isinstance(i, str) for i in user_input):
            pass  # Input is already in the correct format
        else:
            app.logger.error(f"CHAT_ENDPOINT_ERROR: Invalid input format. Type of user_input: {type(user_input)}, Content of user_input: {user_input}")
            return jsonify({"error": "Invalid input format"}), 400

        app.logger.debug(f"CHAT_ENDPOINT - Type of user_input after format check: {type(user_input)}")
        app.logger.debug(f"CHAT_ENDPOINT - Content of user_input after format check: {user_input}")

        # Maintain conversation context
        if user_id not in conversation_context:
            conversation_context[user_id] = []

        conversation_context[user_id].append(user_input)

        # Tokenize the input
        tokenized_input = tokenizer(user_input, padding=True, truncation=True).input_ids

        # Log the type and content of tokenized_input before conversion to JAX array
        app.logger.debug(f"CHAT_ENDPOINT - Type of tokenized_input before JAX conversion: {type(tokenized_input)}")
        app.logger.debug(f"CHAT_ENDPOINT - Content of tokenized_input before JAX conversion: {tokenized_input}")

        # Ensure tokenized input is in the correct format for the model
        tokenized_input = [list(map(int, token)) for token in tokenized_input]  # Convert tokenized input to List[List[int]]

        # Additional logging to capture the state of user_input and tokenized_input
        app.logger.debug(f"CHAT_ENDPOINT - Type of user_input after JAX conversion: {type(user_input)}")
        app.logger.debug(f"CHAT_ENDPOINT - Content of user_input after JAX conversion: {user_input}")
        app.logger.debug(f"CHAT_ENDPOINT - Type of tokenized_input after JAX conversion: {type(tokenized_input)}")
        app.logger.debug(f"CHAT_ENDPOINT - Content of tokenized_input after JAX conversion: {tokenized_input}")

        app.logger.debug(f"CHAT_ENDPOINT - Dtype of tokenized_input before model call: {jax.numpy.array(tokenized_input).dtype}")
        try:
            app.logger.debug(f"CHAT_ENDPOINT - Calling model with tokenized_input: {tokenized_input}")
            output = transformed_model_fn.apply(params, rng, tokenized_input)
            app.logger.debug(f"Model output: {output}")  # Log the model output for debugging
        except Exception as model_call_error:
            app.logger.error(f"Error during model call: {model_call_error}")
            app.logger.error(f"Stack trace: {traceback.format_exc()} - TIMESTAMP: {datetime.now().isoformat()}")
            raise
        try:
            app.logger.debug(f"CHAT_ENDPOINT - Decoding model output: {output}")
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            app.logger.debug(f"Decoded response: {response}")  # Log the decoded response for debugging
        except Exception as decode_error:
            app.logger.error(f"Error during response decoding: {decode_error}")
            app.logger.error(f"Stack trace: {traceback.format_exc()} - TIMESTAMP: {datetime.now().isoformat()}")
            raise
        # Update conversation context with the response
        conversation_context[user_id].append(response)

        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"CHAT_ENDPOINT_ERROR: {e} - TIMESTAMP: {datetime.now().isoformat()}")
        app.logger.error(f"Stack trace: {traceback.format_exc()} - TIMESTAMP: {datetime.now().isoformat()}")
        app.logger.error(f"sys.path during error: {sys.path} - TIMESTAMP: {datetime.now().isoformat()}")
        app.logger.error(f"Environment PATH during error: {os.environ['PATH']} - TIMESTAMP: {datetime.now().isoformat()}")
        app.logger.error(f"Environment PYTHONPATH during error: {os.environ.get('PYTHONPATH', '')} - TIMESTAMP: {datetime.now().isoformat()}")
        try:
            python_interpreter = sys.executable
            installed_packages = subprocess.check_output([python_interpreter, '-m', 'pip', 'list']).decode('utf-8')
            app.logger.error(f"Python interpreter: {python_interpreter} - TIMESTAMP: {datetime.now().isoformat()}")
            app.logger.error(f"Installed packages: {installed_packages} - TIMESTAMP: {datetime.now().isoformat()}")
        except Exception as pkg_error:
            app.logger.error(f"Error retrieving installed packages: {pkg_error} - TIMESTAMP: {datetime.now().isoformat()}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/env', methods=['GET'])
def get_env():
    return jsonify({
        "sys.path": sys.path,
        "PATH": os.environ['PATH'],
        "PYTHONPATH": os.environ.get('PYTHONPATH', '')
    })

if __name__ == '__main__':
    print(f"sys.path at startup: {sys.path}")  # Print sys.path at startup for debugging
    print(f"Environment PATH at startup: {os.environ['PATH']}")  # Print the PATH environment variable at startup for debugging
    print(f"Environment PYTHONPATH at startup: {os.environ.get('PYTHONPATH', '')}")  # Print the PYTHONPATH environment variable at startup for debugging
    app.run(host='0.0.0.0', port=5000)