# VishwamAI Chat Agent

## Overview

VishwamAI is an advanced virtual assistant chat model designed to achieve high accuracy on the MMLU benchmark, with a focus on mathematical reasoning and other benchmarks such as HellaSwag. The model leverages powerful tokenization capabilities and a unique architecture to excel in various reasoning tasks.

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- TensorFlow 2.16.1
- JAX
- Haiku
- Keras NLP
- Optax

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VishwamAI/chat-agent.git
   cd chat-agent/VishwamAI
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install additional dependencies:
   ```bash
   pip install tensorflow keras_nlp jax haiku optax
   ```

### Configuration

Update the `config.py` file with the appropriate paths and settings:
```python
# config.py

VOCAB_FILE = "/home/ubuntu/chat-agent/VishwamAI/data/vishwamai.spm"
RNG_SEED = 42
```

## Usage

### Training the Model

To train the VishwamAI model, use the `train_vishwamai_model.py` script:
```bash
python scripts/train_vishwamai_model.py
```

### Evaluating the Model

To evaluate the VishwamAI model's performance on MMLU and math reasoning benchmarks, use the `test_vishwamai_performance.py` script:
```bash
python scripts/test_vishwamai_performance.py
```

### Tokenizer

To train the SentencePiece tokenizer, use the `train_sentencepiece_tokenizer.py` script:
```bash
python scripts/train_sentencepiece_tokenizer.py
```

To clean the SentencePiece model file, use the `clean_spm.py` script:
```bash
python scripts/clean_spm.py
```

## Model Architecture

The VishwamAI model includes the following components:
- Tokenization using SentencePiece
- Transformer application
- Expert networks
- Gating networks
- Output layers

### Example Usage

```python
from model_architecture import VishwamAIModel
import tensorflow as tf
import keras_nlp
import config

# Initialize tokenizer
tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=config.VOCAB_FILE)

# Initialize model
vocab_size = 10000
embed_dim = 128
num_experts = 4
model = VishwamAIModel(vocab_size, embed_dim, num_experts)

# Load model parameters
model.load_weights("vishwamai_model_params.pkl")

# Tokenize input
input_text = "What is the capital of France?"
tokenized_input = tokenizer.tokenize(input_text)
tokenized_input = tf.convert_to_tensor(tokenized_input, dtype=tf.int32)

# Process input through the model
output = model(tokenized_input)

# Decode output
decoded_output = tokenizer.detokenize(tf.argmax(output, axis=-1).numpy())
print(f"Model output: {decoded_output}")
```

## Contributing

Contributions are welcome! Please follow the standard GitHub workflow for contributing to this project.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License.
