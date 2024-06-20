import sys
import os
import jax
import jax.numpy as jnp
from transformers import GPT2Tokenizer
import haiku as hk

# Add the grok_1 subdirectory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'grok-1', 'grok_1')))

# Print the current working directory and Python path for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

from grok_1.model import LanguageModelConfig, TransformerConfig
from grok_1.runners import InferenceRunner, ModelRunner, sample_from_model
from gemma.model import GemmaModel
from model_architecture import VishwamAIModel

def test_model_integration():
    # Initialize the model inside an hk.transform
    def forward_fn(inputs):
        return VishwamAIModel()(inputs)

    transformed_model = hk.transform(forward_fn)
    rng = jax.random.PRNGKey(42)
    params = transformed_model.init(rng, jnp.array([[0]]))  # Correct input shape

    # Define example input
    example_input = "What is the capital of France?"

    # Tokenize and batch the input
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_input = tokenizer.encode(example_input, return_tensors="np")
    batched_input = jnp.array(tokenized_input, dtype=jnp.int32)

    # Process the input through the model
    output = transformed_model.apply(params, rng, batched_input)

    # Print the output
    print(f"Model output: {output}")

    # Verify the output shape and type
    assert isinstance(output, jnp.ndarray), "Output is not a JAX numpy array"
    assert output.shape[-1] == 512, "Output shape is incorrect"

def test_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    example_input = "What is the capital of France?"
    tokenized_input = tokenizer.encode(example_input, return_tensors="np")
    print(f"Tokenized input: {tokenized_input}")
    assert isinstance(tokenized_input, np.ndarray), "Tokenized input is not a numpy array"

def test_transformer():
    def forward_fn(inputs):
        model = VishwamAIModel()
        return model.transformer.apply(model.transformer.init(jax.random.PRNGKey(42), inputs), jax.random.PRNGKey(42), inputs)

    transformed_model = hk.transform(forward_fn)
    rng = jax.random.PRNGKey(42)
    params = transformed_model.init(rng, jnp.array([[0]]))  # Correct input shape

    # Define example input
    example_input = "What is the capital of France?"

    # Tokenize and batch the input
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_input = tokenizer.encode(example_input, return_tensors="np")
    batched_input = jnp.array(tokenized_input, dtype=jnp.int32)

    # Process the input through the model
    output = transformed_model.apply(params, rng, batched_input)

    # Print the output
    print(f"Transformer output: {output}")

    # Verify the output shape and type
    assert isinstance(output, jnp.ndarray), "Output is not a JAX numpy array"
    assert output.shape[-1] == 512, "Output shape is incorrect"

def test_gating_network():
    def forward_fn(inputs):
        model = VishwamAIModel()
        embedded_inputs = model.transformer.apply(model.transformer.init(jax.random.PRNGKey(42), inputs), jax.random.PRNGKey(42), inputs)
        return model.gating_network(embedded_inputs)

    transformed_model = hk.transform(forward_fn)
    rng = jax.random.PRNGKey(42)
    params = transformed_model.init(rng, jnp.array([[0]]))  # Correct input shape

    # Define example input
    example_input = "What is the capital of France?"

    # Tokenize and batch the input
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_input = tokenizer.encode(example_input, return_tensors="np")
    batched_input = jnp.array(tokenized_input, dtype=jnp.int32)

    # Process the input through the model
    output = transformed_model.apply(params, rng, batched_input)

    # Print the output
    print(f"Gating network output: {output}")

    # Verify the output shape and type
    assert isinstance(output, jnp.ndarray), "Output is not a JAX numpy array"

def test_experts():
    def forward_fn(inputs):
        model = VishwamAIModel()
        embedded_inputs = model.transformer.apply(model.transformer.init(jax.random.PRNGKey(42), inputs), jax.random.PRNGKey(42), inputs)
        gate_values = model.gating_network(embedded_inputs)
        expert_indices = jnp.argmax(gate_values, axis=1)
        expert_outputs = []
        for i, expert in enumerate(model.experts):
            mask = (expert_indices == i)
            if jnp.any(mask):
                mask = jnp.expand_dims(mask, axis=-1)  # Expand dimensions of mask to match inputs
                expert_inputs = jnp.where(mask, inputs, 0)  # Ensure expert_inputs are integer dtype
                expert_rng = jax.random.PRNGKey(42)
                expert_params = expert.init(expert_rng, expert_inputs)  # Initialize expert parameters
                expert_output = expert.apply(expert_params, expert_rng, expert_inputs)  # Use apply method
                expert_outputs.append(expert_output)
        return jnp.sum(jnp.stack(expert_outputs), axis=0)

    transformed_model = hk.transform(forward_fn)
    rng = jax.random.PRNGKey(42)
    params = transformed_model.init(rng, jnp.array([[0]]))  # Correct input shape

    # Define example input
    example_input = "What is the capital of France?"

    # Tokenize and batch the input
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_input = tokenizer.encode(example_input, return_tensors="np")
    batched_input = jnp.array(tokenized_input, dtype=jnp.int32)

    # Process the input through the model
    output = transformed_model.apply(params, rng, batched_input)

    # Print the output
    print(f"Experts output: {output}")

    # Verify the output shape and type
    assert isinstance(output, jnp.ndarray), "Output is not a JAX numpy array"

def test_advanced_features():
    def forward_fn(inputs):
        model = VishwamAIModel()
        embedded_inputs = model.transformer.apply(model.transformer.init(jax.random.PRNGKey(42), inputs), jax.random.PRNGKey(42), inputs)
        return model.advanced_features(embedded_inputs)

    transformed_model = hk.transform(forward_fn)
    rng = jax.random.PRNGKey(42)
    params = transformed_model.init(rng, jnp.array([[0]]))  # Correct input shape

    # Define example input
    example_input = "What is the capital of France?"

    # Tokenize and batch the input
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_input = tokenizer.encode(example_input, return_tensors="np")
    batched_input = jnp.array(tokenized_input, dtype=jnp.int32)

    # Process the input through the model
    output = transformed_model.apply(params, rng, batched_input)

    # Print the output
    print(f"Advanced features output: {output}")

    # Verify the output shape and type
    assert isinstance(output, jnp.ndarray), "Output is not a JAX numpy array"

if __name__ == "__main__":
    test_model_integration()
    test_tokenizer()
    test_transformer()
    test_gating_network()
    test_experts()
    test_advanced_features()
