import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def integrate_flan_t5():
    try:
        # Load the FLAN-T5-XXL tokenizer and model
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    try:
        # Example input text
        input_text = "translate English to German: How old are you?"

        # Tokenize the input text
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        # Generate output using the FLAN-T5-XXL model
        outputs = model.generate(input_ids)

        # Decode the output into human-readable text
        generated_text = tokenizer.decode(outputs[0])

        print("Generated Text:", generated_text)
    except Exception as e:
        print(f"Error during text generation: {e}")

if __name__ == "__main__":
    integrate_flan_t5()
