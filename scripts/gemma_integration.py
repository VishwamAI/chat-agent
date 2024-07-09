import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text_with_gemma(input_text):
    try:
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-9b",
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )

        # Prepare the input
        input_ids = tokenizer(input_text, return_tensors="pt").to(device)

        # Generate output
        outputs = model.generate(**input_ids)
        generated_text = tokenizer.decode(outputs[0])

        return generated_text

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    input_text = "Write me a poem about Machine Learning."
    generated_text = generate_text_with_gemma(input_text)
    if generated_text:
        print(generated_text)
    else:
        print("Failed to generate text.")
