import os
from huggingface_hub import HfApi

# Get the Hugging Face token from the environment variable
hf_token = os.getenv("Hugging_Face_Hugging_Face")

# Define the file path to the model checkpoint
file_path = "vishwamai_model/vishwamai_model_checkpoint.bin"

# Initialize the Hugging Face API
api = HfApi()

# Upload the file to the Hugging Face repository
response = api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo=os.path.basename(file_path),
    repo_id="VishwamAI/vishwamai-model",
    token=hf_token
)

# Check the response status
if response:
    print("File uploaded successfully.")
else:
    print("Failed to upload file.")
