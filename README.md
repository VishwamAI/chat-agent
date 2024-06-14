# VishwamAI

## Project Overview
VishwamAI is a chat agent model designed to achieve 100% accuracy in MMLU, math, and reasoning tasks. The model incorporates a game-like design where it awards points for correct answers and deducts points for incorrect ones. VishwamAI leverages libraries from Google, OpenAI, and Microsoft, and is implemented using TensorFlow and PyTorch.

## File Structure
```
VishwamAI/
├── data/               # Directory for datasets
├── models/             # Directory for storing trained models
├── scripts/            # Directory for scripts (e.g., training, preprocessing, auto-update)
├── logs/               # Directory for training logs and metrics
├── config/             # Directory for configuration files
├── utils/              # Directory for utility scripts and functions
├── setup.sh            # Script for setting up the environment
├── requirements.txt    # File for specifying required dependencies
└── README.md           # Project overview and instructions
```

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

## Usage
1. **Run the chat agent model:**
   ```bash
   python3 scripts/chat_agent_model.py
   ```

2. **Interact with VishwamAI:**
   - Provide input to the chat agent.
   - Indicate whether the response was correct or incorrect.
   - The scoring system will update the score based on your feedback.

## Auto-Update Feature
The `setup.sh` script is designed to check for updates and install them if available each time it is run. This ensures that the environment and dependencies are always up to date.

## Documentation
The development process and results are documented in various formats, including txt and MD files. Refer to the `chat_agent_model_design.md` file for detailed information on the model's architecture, components, and implementation plan.

## Contributing
Contributions are welcome! Please follow the standard guidelines for contributing to this project.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
