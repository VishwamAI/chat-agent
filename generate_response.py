import torch
from transformers import T5Tokenizer
from models import Transformer, TransformerConfig
import flax.serialization as flax_serialization
import jax
import jax.numpy as jnp
import os
import sys
import time

def generate_response(input_text):
    # Load the trained model and tokenizer
    config = TransformerConfig(
        vocab_size=32128,
        output_vocab_size=32128,
        emb_dim=512,
        num_heads=8,
        num_layers=6,
        qkv_dim=512,
        mlp_dim=2048,
        max_len=512,  # Updated max_len to 512
        dropout_rate=0.3,
        attention_dropout_rate=0.3
    )
    model = Transformer(config=config)
    with open('./vishwam_model/model_params.msgpack', 'rb') as f:
        model_params = flax_serialization.from_bytes(model, f.read())
    model = model.clone()
    model.params = model_params
    tokenizer = T5Tokenizer.from_pretrained('./vishwam_model', local_files_only=True)

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='jax', max_length=512, truncation=True)
    print("Tokenized input:", input_ids)

    # Generate response using the model with beam search
    def generate_step(params, input_ids, beam_size=3):
        logits = model.apply({"params": params}, inputs=input_ids, train=False)
        print("Logits:", logits)
        # Use beam search to select the next token based on the probability distribution
        beam_search_output = jax.lax.top_k(logits, k=beam_size)
        output_ids = beam_search_output[1]  # Get the token IDs from the beam search output
        output_ids = output_ids.flatten()  # Flatten the output IDs
        return output_ids

    output_ids = generate_step(model.params, input_ids)
    print("Output IDs:", output_ids)

    # Decode the generated response
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    return response

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 generate_response.py <input_text>")
        sys.exit(1)
    input_text = sys.argv[1]
    response = generate_response(input_text)
    print("Input:", input_text)
    print("Response:", response)
