from transformers import GPT2Tokenizer
import jax.numpy as jnp

# Initialize the tokenizer and set the padding token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the input without converting to JAX tensors
example_input = ["dummy input"]
tokenized_input = tokenizer(example_input, padding=True, truncation=True).input_ids

# Manually convert the tokenized input to a JAX numpy array
tokenized_input_jax = jnp.array(tokenized_input, dtype=jnp.int32)

# Print the tokenized input and the JAX numpy array
print("Tokenized input:", tokenized_input)
print("Tokenized input (JAX):", tokenized_input_jax)
