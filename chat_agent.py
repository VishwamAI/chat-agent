import requests
from bs4 import BeautifulSoup
from transformers import T5Tokenizer
from models import Transformer, TransformerConfig
import flax.serialization as flax_serialization
import jax
import jax.numpy as jnp
import time
import threading
import concurrent.futures
import json

class AutonomousLearner:
    def __init__(self, model_path='./vishwam_model'):
        self.github_token = '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'
        self.huggingface_token = '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'

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
        try:
            with open('dataset.json', 'r') as f:
                self.dataset = json.load(f)
        except FileNotFoundError:
            print("Error: dataset.json file not found. Please ensure the file exists.")
            self.dataset = []
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from dataset.json. Please check the file format.")
            self.dataset = []

        import language_tool_python
        self.language_tool = language_tool_python.LanguageTool('en-US')

    def __del__(self):
        self.language_tool.close()

    def fetch_web_content(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error: Failed to fetch content from {url}: {e}")
            return ""
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()

    def search_github_repositories(self, query, per_page=10):
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        params = {
            'q': query,
            'per_page': per_page
        }
        response = requests.get('https://api.github.com/search/repositories', headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get('items', [])
        else:
            print(f"Error: Failed to search GitHub repositories: {response.status_code} {response.text}")
            return []

    def search_huggingface_models(self, query, limit=10):
        headers = {
            'Authorization': f'Bearer {self.huggingface_token}',
            'Accept': 'application/json'
        }
        params = {
            'search': query,
            'limit': limit
        }
        response = requests.get('https://huggingface.co/api/models', headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get('models', [])
        else:
            print(f"Error: Failed to search Hugging Face models: {response.status_code} {response.text}")
            return []

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
            sampled_ids = beam_search_output[1]  # Select the top beams
            # Ensure sampled IDs are within the valid range of the tokenizer's vocabulary
            sampled_ids = jnp.clip(sampled_ids, 0, self.tokenizer.vocab_size - 1)
            return sampled_ids

        output_ids = generate_step(self.model.params, input_ids)
        responses = []
        for beam in output_ids:
            # Flatten the output_ids before decoding
            beam = beam.flatten()
            # Decode the output_ids to generate the response
            response = self.tokenizer.decode(beam, skip_special_tokens=True)
            # Correct the grammar of the generated response
            corrected_response = self.correct_grammar(response)
            responses.append(corrected_response)

        # Select the best response based on fluency, relevance, and grammatical correctness
        best_response = max(responses, key=lambda r: self.evaluate_response(r, input_ids))
        return best_response

    def correct_grammar(self, text):
        matches = self.language_tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text

    def evaluate_response(self, response, expected_response):
        # Use semantic similarity to evaluate the response
        similarity = self.calculate_similarity(response, expected_response)
        return similarity > 0.7  # Consider responses with similarity above 0.7 as correct

    def learn_from_web(self, url):
        content = self.fetch_web_content(url)
        input_ids = self.process_content(content)
        response = self.update_model(input_ids)
        return response

    def continuous_retrain(self, urls, interval=3600):
        while True:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.learn_from_web, url) for url in urls]
                for future in concurrent.futures.as_completed(futures):
                    response = future.result()
                    print("Learned Response:", response)
                    expected_response = self.get_expected_response(response)
                    is_correct = self.evaluate_response(response, expected_response)
                    print("Evaluation Result:", is_correct)
                    self.update_dataset(response, expected_response)

                # Search GitHub repositories and Hugging Face models
                github_repos = self.search_github_repositories("chatbot")
                huggingface_models = self.search_huggingface_models("chatbot")

                # Process the retrieved information
                for repo in github_repos:
                    repo_content = self.fetch_web_content(repo['html_url'])
                    repo_input_ids = self.process_content(repo_content)
                    repo_response = self.update_model(repo_input_ids)
                    self.update_dataset(repo_content, repo_response)

                for model in huggingface_models:
                    model_content = self.fetch_web_content(model['modelId'])
                    model_input_ids = self.process_content(model_content)
                    model_response = self.update_model(model_input_ids)
                    self.update_dataset(model_content, model_response)

            print(f"Waiting for {interval} seconds before the next iteration...")
            time.sleep(interval)

    def get_expected_response(self, input_text):
        best_match = ""
        highest_similarity = 0
        for pair in self.dataset:
            similarity = self.calculate_similarity(input_text, pair["input"])
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = pair["output"]
        return best_match

    def calculate_similarity(self, text1, text2):
        # Tokenize the texts
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        # Calculate the Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        similarity = len(intersection) / len(union) if union else 0
        return similarity

    def update_dataset(self, input_text, output_text):
        # Check for redundancy
        for pair in self.dataset:
            if pair["input"].strip().lower() == input_text.strip().lower() and pair["output"].strip().lower() == output_text.strip().lower():
                print("Redundant input-output pair. Skipping update.")
                return
        # Add new input-output pair to the dataset
        new_pair = {"input": input_text, "output": output_text}
        self.dataset.append(new_pair)
        # Save the updated dataset to the JSON file
        with open('dataset.json', 'w') as f:
            json.dump(self.dataset, f, indent=4)
        print("Dataset updated with new input-output pair")

if __name__ == "__main__":
    learner = AutonomousLearner()
    try:
        with open('urls.txt', 'r') as f:
            urls = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Error: urls.txt file not found. Please ensure the file exists.")
        urls = []
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading urls.txt: {e}")
        urls = []
    learner.continuous_retrain(urls)
