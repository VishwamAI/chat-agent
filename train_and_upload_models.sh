#!/bin/bash

# Script to train and upload models to Hugging Face

# Ensure all necessary dependencies are installed
pip install -r requirements.txt

# Function to train the model
train_model() {
  model_size=$1
  config_file=$2

  echo "Training Vishwamai model with size ${model_size}..."
  if [ ! -f ${config_file} ]; then
    echo "Error: Configuration file ${config_file} not found."
    exit 1
  fi
  python train_t5.py --config ${config_file} --output_dir ./models/vishwamai-${model_size}
  if [ $? -ne 0 ]; then
    echo "Error: Training failed for model size ${model_size}."
    exit 1
  fi
}

# Function to upload the model to Hugging Face
upload_model() {
  model_size=$1
  repo_name="VishwamAI/vishwamai-${model_size}"

  echo "Uploading Vishwamai model with size ${model_size} to Hugging Face..."
  if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN environment variable is not set."
    exit 1
  fi
  huggingface-cli login --token $HUGGINGFACE_TOKEN
  if [ $? -ne 0 ]; then
    echo "Error: Hugging Face login failed."
    exit 1
  fi
  huggingface-cli repo create ${repo_name} --type model
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create repository ${repo_name}."
    exit 1
  fi
  huggingface-cli upload ./models/vishwamai-${model_size} --repo ${repo_name}
  if [ $? -ne 0 ]; then
    echo "Error: Failed to upload model size ${model_size} to Hugging Face."
    exit 1
  fi
}

# Train and upload models
train_model "2b" "config_for_2b.yaml"
upload_model "2b"

train_model "7b" "config_for_7b.yaml"
upload_model "7b"

train_model "9b" "config_for_9b.yaml"
upload_model "9b"

train_model "27b" "config_for_27b.yaml"
upload_model "27b"

echo "All models have been trained and uploaded successfully."
