import flax.serialization as flax_serialization
from models import Transformer, TransformerConfig

def check_model_params():
    config = TransformerConfig(
        vocab_size=32128,
        output_vocab_size=32128,
        emb_dim=512,
        num_heads=8,
        num_layers=6,
        qkv_dim=512,
        mlp_dim=2048,
        max_len=2048,
        dropout_rate=0.3,
        attention_dropout_rate=0.3
    )
    model = Transformer(config=config)
    try:
        with open('./vishwam_model/model_params.msgpack', 'rb') as f:
            model_params = flax_serialization.from_bytes(model, f.read())
        model = model.clone()
        model.params = model_params
        print('Model parameters loaded successfully.')
    except Exception as e:
        print(f'Error loading model parameters: {e}')

if __name__ == "__main__":
    check_model_params()
