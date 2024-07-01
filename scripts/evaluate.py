import os
import sys
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
from transformers import AutoTokenizer
from src.model.architecture import VishwamAILLM
from src.training.trainer import VishwamAITrainer
from src.data.dataset import create_dataset
from scripts.bias_analysis import analyze_bias
import yaml
from sklearn.metrics import accuracy_score, f1_score

# Add the parent directory to the system path to resolve the import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def evaluate_model():
    # Load configuration
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs/default_config.yaml'))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    config['pad_token_id'] = config.get('pad_token_id', tokenizer.pad_token_id)

    # Add mathematical symbols to tokenizer
    math_symbols = ['+', '-', '*', '/', '=', '(', ')', '^', 'sqrt', 'pi', 'e']
    tokenizer.add_tokens(math_symbols)

    # Create evaluation dataset
    eval_dataset = create_dataset(config['eval_file'], tokenizer, config['batch_size'], config['max_seq_length'])

    # Initialize model
    def model_fn(inputs):
        model = VishwamAILLM(config)
        return model(inputs, is_training=False)

    model = hk.transform(model_fn)

    # Load trained parameters
    checkpoint_path = '/home/ubuntu/chat-agent/VishwamAI-main/checkpoints/model_checkpoint.npy'
    trained_params = np.load(checkpoint_path, allow_pickle=True).item()

    # Evaluate model
    print("Evaluating model...")
    all_predictions = []
    all_labels = []
    for batch in eval_dataset:
        outputs = model.apply(trained_params, None, batch['input_ids'])
        predictions = jnp.argmax(outputs, axis=-1)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        all_predictions.extend(decoded_predictions)
        all_labels.extend(decoded_labels)
        for prediction in decoded_predictions:
            print(f"Model Prediction: {prediction}")

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Analyze model outputs for biases
    print("Analyzing model outputs for biases...")
    for batch in eval_dataset:
        text_batch = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        for text in text_batch:
            bias_results = analyze_bias(text)
            print(f"Bias Analysis Results for model outputs: {bias_results}")

if __name__ == "__main__":
    evaluate_model()
