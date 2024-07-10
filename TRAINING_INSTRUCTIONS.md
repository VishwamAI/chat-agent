# Training Instructions for Vishwamai Model

## Prerequisites

1. **Python Environment**: Ensure you have Python 3.8 or later installed.
2. **Virtual Environment**: It is recommended to use a virtual environment to manage dependencies.
3. **Git LFS**: Ensure Git LFS is installed and initialized in the repository.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VishwamAI/chat-agent.git
   cd chat-agent
   ```

2. **Initialize Git LFS**:
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

Ensure the dataset is available in the `datasets` directory. The `config_for_9b.yaml` file is configured to use `datasets/dev.json` for both training and validation.

## Training the Model

1. **Run the Training Script**:
   ```bash
   python scripts/train_t5.py --config configs/config_for_9b.yaml
   ```

## Notes

- The `train_t5.py` script is designed to train the Vishwamai model using the specified configuration file.
- Ensure that the `datasets/dev.json` file is correctly formatted and available in the `datasets` directory.
- The training process may take a significant amount of time, depending on the size of the dataset and the available computational resources.

## Troubleshooting

- If you encounter any issues with missing dependencies, ensure that all required packages are listed in the `requirements.txt` file and installed in your virtual environment.
- For any other issues, refer to the repository's README file or seek assistance from the repository maintainers.

This PR was written by [Devin](https://devin.ai/) :angel:
