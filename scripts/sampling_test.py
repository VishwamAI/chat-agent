import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

def generate_response(prompt, temperature=0.95, top_p=1.0, top_k=100):
    try:
        # Authenticate with Hugging Face using environment variable
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token not found in environment variables.")
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
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate output with specified sampling parameters
        outputs = model.generate(
            input_ids=input_ids['input_ids'],
            max_length=100,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_response

    except Exception as e:
        print(f"Error generating response: {e}")
        return None

if __name__ == "__main__":
    prompts = [
        "Hi, how can I assist you today?",
        "What is the capital of France?",
        "Tell me a joke.",
        "Explain the theory of relativity."
    ]
    temperatures = [0.7, 0.9, 1.0]
    top_ps = [0.8, 0.9, 1.0]
    top_ks = [50, 100, 200]

    for prompt in prompts:
        for temperature in temperatures:
            for top_p in top_ps:
                for top_k in top_ks:
                    print(f"Prompt: {prompt}")
                    print(f"Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}")
                    response = generate_response(prompt, temperature, top_p, top_k)
                    if response:
                        print(f"Response: {response}\n")
                    else:
                        print("Failed to generate response.\n")
