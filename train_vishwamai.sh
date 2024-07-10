#!/bin/bash

# Script to train the Vishwamai model using the train_t5.py script

# Ensure the script is executed from the root of the repository
cd "$(dirname "$0")"

# Activate the virtual environment
source venv/bin/activate

# Initialize Git LFS
git lfs install
git lfs pull

# Run the training script with the specified configuration
python scripts/train_t5.py --dataset datasets/dev.json --model_size 9b --epochs 3 --batch_size 8

# Deactivate the virtual environment
deactivate
