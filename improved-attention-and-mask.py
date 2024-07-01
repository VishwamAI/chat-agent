import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Optional

class ImprovedAttention(hk.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.num_heads = config['num_heads']
        self.head_dim = config['embed_dim'] // config['num_heads']
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[Dict] = None):
        seq_len = x.shape[1]
        qkv = hk.Linear(3 * self.num_heads * self.head_dim, with_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(-1, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(-1, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(-1, seq_len, self.num_heads, self.head_dim)

        sincos = self.rotary_emb(seq_len)
        q = apply_rotary_pos_emb(q, sincos)
        k = apply_rotary_pos_emb(k, sincos)

        if kv_cache is not None:
            if kv_cache['k'] is None:
                kv_cache['k'] = k
                kv_cache['v'] = v
            else:
                k = jnp.concatenate([kv_cache['k'], k], axis=1)
                v = jnp.concatenate([kv_cache['v'], v], axis=1)
                kv_cache['k'] = k
                kv_cache['v'] = v

        attn = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            # Ensure mask shape matches attention tensor's shape
            mask = jnp.broadcast_to(mask[:, None, None, :], (mask.shape[0], self.num_heads, seq_len, k.shape[1]))
            attn = jnp.where(mask, attn, float('-inf'))

        attn = jax.nn.softmax(attn, axis=-1)

        output = jnp.einsum('bhqk,bkhd->bqhd', attn, v)
        return output.reshape(-1, seq_len, self.num_heads * self.head_dim)

class ImprovedVishwamAIModel(hk.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.embed_dim = config['embed_dim']
        self.num_layers = config['num_layers']
        self.vocab_size = config['vocab_size']

    def __call__(self, inputs: jnp.ndarray, is_training: bool = False, kv_cache: Optional[Dict] = None) -> jnp.ndarray:
        mask = self._create_mask(inputs)
        x = self._embed(inputs)

        if kv_cache is None:
            kv_cache = [{'k': None, 'v': None} for _ in range(self.num_layers)]

        for i in range(self.num_layers):
            x = ImprovedTransformerBlock(self.config)(x, mask, kv_cache[i], is_training)

        return hk.Linear(self.vocab_size)(x), kv_cache

    def _create_mask(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if self.config['pad_token_id'] is None:
            raise ValueError("pad_token_id is not set in the configuration.")
        
        # Create padding mask
        padding_mask = jnp.not_equal(inputs, self.config['pad_token_id']).astype(jnp.float32)
        
        # Create causal mask
        seq_length = inputs.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=jnp.float32))
        
        # Combine padding mask and causal mask
        mask = padding_mask[:, None] * causal_mask[None, :]
        
        return mask

# Test function to verify mask creation and attention mechanism
def test_model():
    config = {
        'num_heads': 8,
        'embed_dim': 512,
        'ff_dim': 2048,
        'num_layers': 6,
        'vocab_size': 50000,
        'dropout_rate': 0.1,
        'pad_token_id': 0,
    }

    def forward_pass(inputs):
        model = ImprovedVishwamAIModel(config)
        return model(inputs)

    model = hk.transform(forward_pass)
    
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, jnp.ones((2, 10), dtype=jnp.int32))
    
    # Test with different sequence lengths
    for seq_len in [10, 20, 30]:
        inputs = jax.random.randint(rng, (2, seq_len), 0, config['vocab_size'])
        outputs, _ = model.apply(params, rng, inputs)
        print(f"Output shape for sequence length {seq_len}: {outputs.shape}")

# Run the test
test_model()