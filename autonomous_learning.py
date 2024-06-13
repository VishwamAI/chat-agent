import requests
from bs4 import BeautifulSoup
from transformers import T5Tokenizer
from models import Transformer, TransformerConfig
import flax.serialization as flax_serialization
import jax
import jax.numpy as jnp
import time

class AutonomousLearner:
    def __init__(self, model_path='./vishwam_model'):
        config = TransformerConfig(
            vocab_size=32128,
            output_vocab_size=32128,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=512,  # Adjusted to match the error message
            dropout_rate=0.3,
            attention_dropout_rate=0.3
        )
        self.model = Transformer(config=config)
        with open(f"{model_path}/model_params.msgpack", 'rb') as f:
            model_params = flax_serialization.from_bytes(self.model, f.read())
        self.model = self.model.clone()
        self.model.params = model_params
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)

    def fetch_web_content(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()

    def process_content(self, content):
        # Tokenize the content
        input_ids = self.tokenizer.encode(content, return_tensors='jax')
        # Ensure the input sequence length does not exceed the model's maximum sequence length
        max_len = 512  # Adjusted to match the error message
        if input_ids.shape[1] > max_len:
            input_ids = input_ids[:, :max_len]
        print("Truncated input length:", input_ids.shape[1])  # Debugging print statement
        return input_ids

    def update_model(self, input_ids):
        # Generate response using the model
        def generate_step(params, input_ids):
            logits = self.model.apply({"params": params}, inputs=input_ids, train=False)
            # Use beam search for better text generation
            beam_size = 5
            beam_search_output = jax.lax.top_k(logits, k=beam_size)
            sampled_ids = beam_search_output[1][0]  # Select the top beam
            # Ensure sampled IDs are within the valid range of the tokenizer's vocabulary
            sampled_ids = jnp.clip(sampled_ids, 0, self.tokenizer.vocab_size - 1)
            return sampled_ids

        output_ids = generate_step(self.model.params, input_ids)
        # Flatten the output_ids before decoding
        output_ids = output_ids.flatten()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response

    def learn_from_web(self, url):
        content = self.fetch_web_content(url)
        input_ids = self.process_content(content)
        response = self.update_model(input_ids)
        return response

    def continuous_retrain(self, urls, interval=3600):
        while True:
            for url in urls:
                print(f"Learning from {url}")
                response = self.learn_from_web(url)
                print("Learned Response:", response)
                # Here you can add code to evaluate the response and update the model if necessary
            print(f"Waiting for {interval} seconds before the next iteration...")
            time.sleep(interval)

if __name__ == "__main__":
    learner = AutonomousLearner()
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]
    learner.continuous_retrain(urls)
