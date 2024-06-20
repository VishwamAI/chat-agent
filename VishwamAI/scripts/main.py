import haiku as hk
import jax
import jax.numpy as jnp
from transformers import GPT2Tokenizer
import requests
import random
from model_architecture import VishwamAIModel
from grok_1.model import LanguageModelConfig, TransformerConfig
from grok_1.runners import InferenceRunner, ModelRunner, sample_from_model

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

    # Initialize Grok-1 model
    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            # MoE.
            num_experts=8,
            num_selected_experts=2,
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )

    inference_runner = InferenceRunner(
        pad_sizes=(1024,),
        runner=ModelRunner(
            model=grok_1_model,
            bs_per_device=0.125,
            checkpoint_path="./checkpoints/",
        ),
        name="local",
        load="./checkpoints/",
        tokenizer_path="./tokenizer.model",
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )

    inference_runner.initialize()
    gen = inference_runner.run()

    inp = "The answer to life the universe and everything is of course"
    print(f"Output for prompt: {inp}", sample_from_model(gen, inp, max_len=100, temperature=0.01))

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
