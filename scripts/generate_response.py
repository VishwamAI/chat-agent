import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

def generate_response(prompt):
    try:
        # Authenticate with Hugging Face
        login(token="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")

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
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate output
        outputs = model.generate(**input_ids)
        generated_response = tokenizer.decode(outputs[0])

        return generated_response

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    prompt = "Hi, how can I assist you today?"
    response = generate_response(prompt)
    if response:
        print(response)
    else:
        print("Failed to generate response.")
