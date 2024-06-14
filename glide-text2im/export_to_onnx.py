import torch
import torch.onnx
from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
from glide_text2im.nn import timestep_embedding

# Define the function to export the model to ONNX format
def export_to_onnx():
    # Set the model options
    options = model_and_diffusion_defaults()
    options['use_fp16'] = False  # Disable FP16 for compatibility
    options['timestep_respacing'] = '100'  # Use 100 diffusion steps for fast sampling

    # Create the GLIDE model and diffusion process
    glide_model, glide_diffusion = create_model_and_diffusion(**options)

    # Load the GLIDE model checkpoint
    checkpoint_path = 'base_checkpoint.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    glide_model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    glide_model.eval()

    # Define dummy input for the model
    dummy_input = torch.randint(0, 256, (1, 3, 16, 16), dtype=torch.float32)  # Further reduced size

    # Define timesteps tensor
    timesteps = torch.tensor([0], dtype=torch.float32)

    # Define tokens tensor
    tokens = torch.randint(0, 1000, (1, 128), dtype=torch.long)  # Reverted tokens tensor size to original

    # Define mask tensor
    mask = torch.ones_like(tokens, dtype=torch.bool)  # Adjusted mask tensor size to match tokens

    # Export the model to ONNX format
    onnx_output_path = 'glide_model.onnx'
    torch.onnx.export(glide_model, (dummy_input, timesteps, tokens, mask), onnx_output_path, opset_version=12)

    print(f"Model has been successfully exported to {onnx_output_path}")

# Run the export function
if __name__ == "__main__":
    export_to_onnx()
