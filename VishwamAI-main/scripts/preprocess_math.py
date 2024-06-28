import os
import sys
import yaml
import jax.numpy as jnp
import haiku as hk
from transformers import AutoTokenizer

# Add the parent directory to the system path to resolve the import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.architecture import MathReasoningLayer

def preprocess_math_expressions(input_file, output_file, config_file):
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-string token
    config['pad_token_id'] = tokenizer.pad_token_id

    # Add mathematical symbols to tokenizer
    math_symbols = ['+', '-', '*', '/', '=', '(', ')', '^', 'sqrt', 'pi', 'e']
    tokenizer.add_tokens(math_symbols)

    # Define a function to initialize and apply MathReasoningLayer
    def apply_math_reasoning_layer(input_ids):
        math_reasoning_layer = MathReasoningLayer(config)
        return math_reasoning_layer(input_ids)

    # Transform the function with Haiku
    transformed_apply = hk.transform(apply_math_reasoning_layer)

    # Read input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Preprocess each line
    preprocessed_lines = []
    for line in lines:
        # Tokenize line
        tokens = tokenizer(line, return_tensors='jax', padding=True, truncation=True)
        input_ids = tokens['input_ids']

        # Apply MathReasoningLayer
        preprocessed_ids = transformed_apply.apply(None, None, input_ids)

        # Convert preprocessed_ids to integer dtype
        preprocessed_ids = jnp.asarray(preprocessed_ids, dtype=jnp.int32)

        # Convert preprocessed_ids back to text
        preprocessed_text = tokenizer.decode(preprocessed_ids[0], skip_special_tokens=True)
        preprocessed_lines.append(preprocessed_text)

    # Write preprocessed lines to output file
    with open(output_file, 'w') as f:
        for line in preprocessed_lines:
            f.write(line + '\n')

if __name__ == "__main__":
    input_file = os.path.join(os.path.dirname(__file__), '../data/raw/train.txt')
    output_file = os.path.join(os.path.dirname(__file__), '../data/processed/train_preprocessed.txt')
    config_file = os.path.join(os.path.dirname(__file__), '../configs/default_config.yaml')
    preprocess_math_expressions(input_file, output_file, config_file)
