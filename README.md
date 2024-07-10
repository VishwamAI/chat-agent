# VishwamAI

VishwamAI is an advanced language model based on the Transformer architecture, designed for various natural language processing tasks.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/VishwamAI/chat-agent.git
   cd chat-agent
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the model

To train the VishwamAI model, run:

```
python scripts/train.py
```

This script will initialize the model and datasets, and start the training process.

### Generating text

To generate text using a trained model, use:

```
python scripts/generate_text.py --prompt "Your prompt here" --max_length 100
```

### Evaluating the model

To evaluate the model on a test dataset, run:

```
python scripts/evaluate.py --test_file path/to/test/file.txt
```

### Testing Sampling Parameters

To test the VishwamAI model with different sampling parameters (temperature, top-p, top-k) across various prompts, use:

```
python scripts/sampling_test.py --prompt "Your prompt here" --temperature 0.7 --top_p 0.9 --top_k 50
```

### Note

Before running any scripts that depend on the Hugging Face API, ensure that the `HUGGING_FACE_TOKEN` environment variable is set up as described in the "Setting Up Environment Variables" section.

## Configuration

You can modify the model and training configuration by editing the configuration files in the `configs/` directory.

## Setting Up Environment Variables

To securely use the Hugging Face API, set the `HUGGING_FACE_TOKEN` environment variable with your Hugging Face token. This can be done by adding the following line to your shell profile (e.g., `.bashrc`, `.zshrc`):

```
export HUGGING_FACE_TOKEN=your_hugging_face_token
```

After adding the line, reload your shell profile:

```
source ~/.bashrc  # or source ~/.zshrc
```

## Documentation

For more detailed information about the model architecture, training process, and API reference, please refer to the `docs/` directory.

## Contributing

Contributions to VishwamAI are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.
