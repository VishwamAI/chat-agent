import numpy as np

def inspect_checkpoint(checkpoint_path):
    try:
        params = np.load(checkpoint_path, allow_pickle=True)
        print(f"Loaded parameters type: {type(params)}")
        if isinstance(params, dict):
            print("Parameters are in the expected dictionary format.")
        else:
            print("Parameters are NOT in the expected dictionary format.")
        print("Parameters content:")
        print(params)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    checkpoint_path = "/home/ubuntu/chat-agent/VishwamAI-main/checkpoints/model_checkpoint.npy"
    inspect_checkpoint(checkpoint_path)
