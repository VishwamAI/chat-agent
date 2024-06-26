# VishwamAI

## Overview
VishwamAI is an advanced virtual assistant chat model designed to achieve 100% accuracy in the MMLU benchmark, with a strong focus on mathematical reasoning and other benchmarks such as HellaSwag. The model leverages libraries from Google, OpenAI, and Microsoft, and is implemented using TensorFlow and JAX. VishwamAI aims to excel in various reasoning tasks and provide accurate and reliable responses.

## File Structure
```
VishwamAI/
├── data/               # Directory for datasets
├── models/             # Directory for storing trained models
├── scripts/            # Directory for scripts (e.g., training, preprocessing, model conversion, auto-update)
├── notebooks/          # Directory for Jupyter notebooks
├── logs/               # Directory for training logs and metrics
├── docs/               # Directory for documentation
├── config/             # Directory for configuration files
├── utils/              # Directory for utility scripts and functions
├── setup.sh            # Script for setting up the environment
├── requirements.txt    # File for specifying required dependencies
└── README.md           # Project overview and instructions
```

## Components
1. **Generative Model**: The core of VishwamAI is a Generative Adversarial Network (GAN) responsible for generating images.
2. **Natural Language Processing (NLP) Component**: This component handles chat interactions, understanding user inputs, generating appropriate responses using a transformer-based model like GPT, and generating questions based on input text.
3. **Self-Improvement Mechanism**: VishwamAI includes mechanisms for self-tuning and improvement, leveraging internet resources for continuous learning and enhancement.

## High-Level Architecture
1. **Input Handling**: The NLP component processes user inputs and converts them into a format suitable for the generative model.
2. **Image Generation**: The GAN creates images based on the processed inputs.
3. **Output Handling**: The generated images are returned to the user along with any relevant textual responses.
4. **Self-Improvement**: The model periodically accesses internet resources to gather new data and improve its performance.
5. **Question Generation**: The NLP component generates questions based on input text to guide further data collection or model training.

## Development Steps
1. **Set Up Development Environment**: Install necessary libraries and dependencies, including TensorFlow and `glide_text2im`.
2. **Design Model Architecture**: Define the structure of the GAN and the NLP component.
3. **Collect and Preprocess Data**: Gather datasets for training the model and preprocess them as needed.
4. **Implement Model**: Code the model architecture and training loop.
5. **Train Model**: Train the model on the collected datasets and monitor its performance.
6. **Evaluate and Improve**: Evaluate the model's performance and make necessary adjustments. Implement self-improvement mechanisms and question generation capabilities.

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/VishwamAI/chat-agent
   cd chat-agent/VishwamAI
   ```

2. **Run the setup script to install dependencies and set up the environment:**
   ```bash
   ./setup.sh
   ```

## Installation
To install the VishwamAI project as an editable package, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/VishwamAI.git
   cd VishwamAI
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the VishwamAI project as an editable package:
   ```bash
   pip install -e .
   ```

## Usage
1. **Run the chat agent model:**
   ```bash
   python3 scripts/chat_agent_model.py
   ```

2. **Interact with VishwamAI:**
   - Provide input to the chat agent.
   - Indicate whether the response was correct or incorrect.
   - The scoring system will update the score based on your feedback.

3. Ensure that the virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```

4. Run the `vishwamai` script to start the model:
   ```bash
   vishwamai
   ```

## Training the Model
To train the VishwamAI model on a small sample dataset, follow these steps:

1. Ensure that the virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```

2. Navigate to the `scripts` directory:
   ```bash
   cd VishwamAI/scripts
   ```

3. Run the training script:
   ```bash
   python -c "from vishwamai_prototype import VishwamAI; vishwamai = VishwamAI(batch_size=32); vishwamai.train(epochs=1000, batch_size=32)"
   ```

## Evaluating Model Performance
To evaluate the performance of the VishwamAI model, follow these steps:

1. Ensure that the virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```

2. Navigate to the `scripts` directory:
   ```bash
   cd VishwamAI/scripts
   ```

3. Run the evaluation script:
   ```bash
   python -c "from vishwamai_prototype import VishwamAI; vishwamai = VishwamAI(batch_size=32); performance_metrics = vishwamai.evaluate_performance(); print(performance_metrics)"
   ```

## Running Unit Tests
To run the unit tests for the VishwamAI project, follow these steps:

1. Ensure that the virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```

2. Navigate to the `scripts` directory:
   ```bash
   cd VishwamAI/scripts
   ```

3. Run the test script:
   ```bash
   python test_vishwamai_prototype.py
   ```

**Note**: The unit tests have been updated to handle cases where the `generate_image` function returns `None`. If an image cannot be generated, an error will be logged, and the test will continue.

## Running Logging Test
To verify the logging output for the VishwamAI project, follow these steps:

1. Ensure that the virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```

2. Navigate to the `scripts` directory:
   ```bash
   cd VishwamAI/scripts
   ```

3. Run the logging test function:
   ```bash
   python -c "from vishwamai_prototype import test_logging; test_logging()"
   ```

**Note**: The logging configuration has been set up to capture and store error messages, including cases where the `generate_image` function returns `None`.

## Contributing
We welcome contributions to the VishwamAI project! If you would like to contribute, please follow these steps:
1. Fork the repository on GitHub.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and descriptive commit messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository with a description of your changes.

**Image Generation Issues**: If the `generate_image` function returns `None`, ensure that the TensorFlow checkpoint file path is correct and accessible. Check the logs for error messages related to the model loading process.

## Troubleshooting
If you encounter any issues while setting up or using the VishwamAI project, here are some common solutions:
- **ModuleNotFoundError**: Ensure that all dependencies are installed and the virtual environment is activated.
- **Virtual Environment Activation**: Make sure you have activated the virtual environment using `source venv/bin/activate`.
- **Missing `glide_text2im` Module**: Clone the official GitHub repository and install it as an editable package.

## Dependencies
The VishwamAI project requires the following dependencies:
- tensorflow
- torch
- onnx
- onnx2keras
- transformers
- tf-keras
- Pillow
- numpy
- scipy
- requests
- hiku
- jax
## Auto-Update Feature
The `setup.sh` script is designed to check for updates and install them if available each time it is run. This ensures that the environment and dependencies are always up to date.

## Documentation
The development process and results are documented in various formats, including txt and MD files. Refer to the `chat_agent_model_design.md` file for detailed information on the model's architecture, components, and implementation plan.

## Contributing
Contributions are welcome! Please follow the standard guidelines for contributing to this project.
These dependencies are listed in the `requirements.txt` file and can be installed using `pip install -r requirements.txt`.
