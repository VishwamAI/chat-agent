# Vishwamai Model Training Guide

This guide provides step-by-step instructions for training the Vishwamai models of sizes 2b, 7b, 9b, and 27b. Follow these steps to set up the training environment, execute the training process, and upload the trained models to Hugging Face.

## Prerequisites

1. Ensure you have Python 3.8 or later installed.
2. Install the required dependencies listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration Files

The configuration files for the Vishwamai models are located in the `chat-agent` directory:
- `config_for_2b.yaml`
- `config_for_7b.yaml`
- `config_for_9b.yaml`
- `config_for_27b.yaml`

Update the `train_dataset` and `validation_dataset` paths in each configuration file with the actual locations of your datasets.

## Training Script

Use the `train_and_upload_models.sh` script to train the models and upload them to Hugging Face. The script includes error handling for various steps in the training and uploading process.

### Training and Uploading Models

1. Open the `train_and_upload_models.sh` script and review the commands.
2. Execute the script to start the training process:
   ```bash
   ./train_and_upload_models.sh
   ```

### Script Overview

The `train_and_upload_models.sh` script performs the following steps:
1. Sets up the training environment.
2. Trains the Vishwamai models using the specified configuration files.
3. Evaluates the models using the specified metrics (Perplexity, BLEU, ROUGE).
4. Uploads the trained models to Hugging Face.

## Uploading Models to Hugging Face

After training, the models will be uploaded to the Hugging Face repository. Ensure you have the necessary authentication tokens and permissions to upload the models.

### Authentication

Set up your Hugging Face authentication token:
1. Create a `.huggingface` directory in your home directory:
   ```bash
   mkdir -p ~/.huggingface
   ```
2. Create a `token` file in the `.huggingface` directory and add your Hugging Face token:
   ```bash
   echo "your_huggingface_token" > ~/.huggingface/token
   ```

### Uploading

The `train_and_upload_models.sh` script includes commands to upload the trained models to Hugging Face. Ensure the script runs successfully to complete the upload process.

## Conclusion

By following this guide, you will be able to train the Vishwamai models of sizes 2b, 7b, 9b, and 27b and upload them to Hugging Face. Ensure all configuration files are updated with the correct dataset paths and that you have the necessary authentication tokens for Hugging Face.

For any issues or further assistance, refer to the documentation or contact the support team.
