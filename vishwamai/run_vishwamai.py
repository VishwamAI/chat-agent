# Adapted run script for VishwamAI model based on gemma_pytorch's run.py
import argparse
import contextlib
import random

import numpy as np
import torch

# TODO: Import VishwamAI model configuration and model class
from vishwamai import config
from vishwamai import model as vishwamai_model

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

def main(args):
    # Construct the VishwamAI model config.
    model_config = config.get_model_config(args.variant)
    model_config.dtype = "float32" if args.device == "cpu" else "float16"
    model_config.quant = args.quant

    # Seed random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create the VishwamAI model and load the weights.
    device = torch.device(args.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = vishwamai_model.VishwamAIForCausalLM(model_config)
        model.load_weights(args.ckpt)
        model = model.to(device).eval()
        print("Model loading done")

    # Generate the response.
    result = model.generate(args.prompt, device, output_len=args.output_len)

    # Print the prompts and results.
    print('======================================')
    print(f'PROMPT: {args.prompt}')
    print(f'RESULT: {result}')
    print('======================================')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--variant", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output_len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    args = parser.parse_args()

    main(args)