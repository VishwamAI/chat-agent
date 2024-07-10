import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

def generate_text_with_vishwamai(input_text):
    """
    Generate text using the Vishwamai model.

    Parameters:
    input_text (str): The input text prompt for text generation.

    Returns:
    str: The generated text based on the input prompt.
    """
    try:
        # Authenticate with Hugging Face
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable is not set.")
        login(token=hf_token)

        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("VishwamAI/vishwamai-model")
        model = AutoModelForCausalLM.from_pretrained(
            "VishwamAI/vishwamai-model",
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )

        # Prepare the input
        input_ids = tokenizer(input_text, return_tensors="pt").to(device)

        # Generate output
        outputs = model.generate(**input_ids)
        generated_text = tokenizer.decode(outputs[0])

        return generated_text

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    input_text = "Write me a poem about Machine Learning."
    generated_text = generate_text_with_vishwamai(input_text)
    if generated_text:
        print(generated_text)
    else:
        print("Failed to generate text.")
