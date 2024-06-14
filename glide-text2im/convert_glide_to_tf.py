import torch
import onnx
from onnx2keras import onnx_to_keras
from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
from glide_text2im.nn import timestep_embedding

def convert_glide_to_tf():
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
    dummy_input = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.float32)  # Original size

    # Convert the model to ONNX format
    onnx_model_path = 'glide_model.onnx'
    torch.onnx.export(glide_model, dummy_input, onnx_model_path, input_names=['input'], output_names=['output'])

    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)

    # Convert the ONNX model to Keras format
    keras_model = onnx_to_keras(onnx_model, ['input'])

    # Save the converted model
    keras_model.save('glide_model_tf.h5')
    print("Model has been successfully converted and saved as 'glide_model_tf.h5'")

# Run the conversion function
if __name__ == "__main__":
    convert_glide_to_tf()
