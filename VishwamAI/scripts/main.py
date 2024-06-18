import haiku as hk
import jax
import jax.numpy as jnp
from transformers import GPT2Tokenizer
import requests
import random
from model_architecture import VishwamAIModel

class ScoringSystem:
    def __init__(self):
        self.score = 0

    def update_score(self, correct, question_type):
        if correct:
            if question_type == "easy":
                self.score += 1
            elif question_type == "medium":
                self.score += 2
            elif question_type == "hard":
                self.score += 3
        else:
            if question_type == "easy":
                self.score -= 1
            elif question_type == "medium":
                self.score -= 2
            elif question_type == "hard":
                self.score -= 3
        return self.score

def check_internet_connectivity():
    try:
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("Internet connectivity check passed.")
        else:
            print("Internet connectivity check failed.")
    except requests.ConnectionError:
        print("No internet connection available.")
    except requests.Timeout:
        print("The request timed out.")

def model_fn(example_input):
    model = VishwamAIModel()
    return model(example_input)

def main():
    # Check internet connectivity
    check_internet_connectivity()

    # Initialize the scoring system
    scoring_system = ScoringSystem()

    # Example input
    example_input = "What is the capital of France?"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token
    tokenized_input = tokenizer(example_input, return_tensors="jax", padding=True, truncation=True).input_ids
    tokenized_input = jax.numpy.array(tokenized_input, dtype=jnp.int32)  # Ensure inputs are integer dtype for embedding layer

    # Debugging print statements to check dtype
    print(f"Tokenized input dtype before model init: {tokenized_input.dtype}")

    transformed_model_fn = hk.transform(model_fn)
    rng = jax.random.PRNGKey(42)
    params = transformed_model_fn.init(rng, tokenized_input)
    output = transformed_model_fn.apply(params, rng, tokenized_input)
    print(f"Model output: {output}")

    # Example scoring update
    correct = True  # This would be determined by the model's output in a real scenario
    question_type = "medium"  # Example question type
    new_score = scoring_system.update_score(correct, question_type)
    print(f"Updated score: {new_score}")

    # Self-improvement example
    model = VishwamAIModel()
    model.self_improve()

if __name__ == "__main__":
    main()
