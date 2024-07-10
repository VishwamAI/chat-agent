#!/bin/bash

# Script to prepare the training dataset for the Vishwamai model

# Directory containing the datasets
DATASETS_DIR="datasets"

# Configuration file to update
CONFIG_FILE="config_for_9b.yaml"

# Check if the datasets directory exists
if [ ! -d "$DATASETS_DIR" ]; then
  echo "Error: The datasets directory '$DATASETS_DIR' does not exist."
  exit 1
fi

# Find JSON files in the datasets directory
TRAIN_DATASET=$(find "$DATASETS_DIR" -name "*.json" | grep -v "dev.json" | head -n 1)

# Check if a training dataset was found
if [ -z "$TRAIN_DATASET" ]; then
  echo "Error: No training dataset found in the '$DATASETS_DIR' directory."
  echo "Please provide a training dataset in JSON format and place it in the '$DATASETS_DIR' directory."
  echo "You can download a suitable dataset from a known source or create one from raw data."
  exit 1
fi

# Update the configuration file with the path to the training dataset
sed -i "s|path/to/train_dataset|$TRAIN_DATASET|g" "$CONFIG_FILE"

echo "Training dataset found: $TRAIN_DATASET"
echo "Configuration file '$CONFIG_FILE' has been updated with the training dataset path."
