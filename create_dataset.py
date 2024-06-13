import json

def create_dataset():
    # Define a list of input-output pairs for training
    dataset = [
        {"input": "hi", "output": "hello how can I assist you today"},
        {"input": "what is your name", "output": "I am Vishwam, your chat assistant"},
        {"input": "how are you", "output": "I am just a program, but I am functioning as expected"},
        {"input": "tell me a joke", "output": "Why don't scientists trust atoms? Because they make up everything!"},
        {"input": "what is the weather like", "output": "I am not able to check the weather right now, but I hope it's nice where you are"},
        # Add more input-output pairs as needed
    ]

    # Save the dataset to a JSON file
    with open('dataset.json', 'w') as f:
        json.dump(dataset, f, indent=4)
    print("Dataset created and saved to dataset.json")

if __name__ == "__main__":
    create_dataset()
