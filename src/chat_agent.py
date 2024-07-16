import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class ChatAgent:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def chat(self, input_text):
        # Encode the input text
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

        # Generate a response using the model
        output = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)

        # Decode the response
        response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response_text

# Example usage
if __name__ == "__main__":
    model_name = "VishwamAI/vishwamai"  # Placeholder for the actual model name
    chat_agent = ChatAgent(model_name)

    while True:
        input_text = input("You: ")
        if input_text.lower() == "quit":
            break
        response = chat_agent.chat(input_text)
        print(f"ChatAgent: {response}")