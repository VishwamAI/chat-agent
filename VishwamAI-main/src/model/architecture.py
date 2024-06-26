import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Optional,Tuple
from functools import partial

def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(hk.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, seq_len):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2) / self.dim))
        t = jnp.arange(seq_len)
        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.sin(emb), jnp.cos(emb)

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
            k = jnp.concatenate([kv_cache['k'], k], axis=1)
            v = jnp.concatenate([kv_cache['v'], v], axis=1)
            kv_cache['k'] = k
            kv_cache['v'] = v

        attn = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            attn = jnp.where(mask[:, None, None, :], attn, float('-inf'))
        attn = jax.nn.softmax(attn, axis=-1)

        output = jnp.einsum('bhqk,bkhd->bqhd', attn, v)
        return output.reshape(-1, seq_len, self.num_heads * self.head_dim)

class ImprovedTransformerBlock(hk.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.attention = ImprovedAttention(config)
        self.feed_forward = hk.Sequential([
            hk.Linear(config['ff_dim']),
            jax.nn.gelu,
            hk.Linear(config['embed_dim']),
        ])
        self.layer_norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layer_norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.dropout = lambda x: hk.dropout(hk.next_rng_key(), config['dropout_rate'], x)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[Dict] = None, is_training: bool = False) -> jnp.ndarray:
        attention_output = self.attention(self.layer_norm1(x), mask, kv_cache)
        attention_output = self.dropout(attention_output) if is_training else attention_output
        x = x + attention_output
        
        ff_output = self.feed_forward(self.layer_norm2(x))
        ff_output = self.dropout(ff_output) if is_training else ff_output
        x = x + ff_output
        return x

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

    def _embed(self, x: jnp.ndarray) -> jnp.ndarray:
        embedding_matrix = hk.get_parameter("embedding_matrix", 
                                            shape=[self.vocab_size, self.embed_dim],
                                            init=hk.initializers.TruncatedNormal(stddev=0.02))
        return jnp.take(embedding_matrix, x, axis=0)

    def _create_mask(self, inputs: jnp.ndarray) -> jnp.ndarray:
        mask = (inputs != self.config['pad_token_id']).astype(jnp.float32)[:, jnp.newaxis, jnp.newaxis, :]
        seq_length = inputs.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_length, seq_length)))
        return mask * causal_mask[jnp.newaxis, jnp.newaxis, :, :]

class VishwamAILLM(hk.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.transformer = ImprovedVishwamAIModel(config)
        self.lm_head = hk.Linear(config['vocab_size'])

    def __call__(self, inputs: jnp.ndarray, is_training: bool = False, kv_cache: Optional[Dict] = None) -> Tuple[jnp.ndarray, Dict]:
        transformer_outputs, new_kv_cache = self.transformer(inputs, is_training, kv_cache)
        lm_logits = self.lm_head(transformer_outputs)
        return lm_logits, new_kv_cache

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def generate_with_evaluation(self, input_ids: jnp.ndarray, kv_cache: Optional[Dict] = None, max_length: int = 100, temperature: float = 1.0) -> Tuple[jnp.ndarray, Dict]:
        generated_ids = input_ids
        total_log_probs = 0.0
        for _ in range(max_length - input_ids.shape[1]):
            logits, kv_cache = self(generated_ids, is_training=False, kv_cache=kv_cache)
            next_token_logits = logits[:, -1, :] / temperature
            next_token_probs = jax.nn.softmax(next_token_logits, axis=-1)
            next_token = jax.random.categorical(hk.next_rng_key(), next_token_logits, axis=-1)
            generated_ids = jnp.concatenate([generated_ids, next_token[:, jnp.newaxis]], axis=-1)
            
            # Calculate log probability of the chosen token
            total_log_probs += jnp.log(next_token_probs[0, next_token[0]])

        # Calculate perplexity
        sequence_length = generated_ids.shape[1] - input_ids.shape[1]
        perplexity = jnp.exp(-total_log_probs / sequence_length)

        # Calculate confidence
        confidence = jnp.mean(jnp.max(jax.nn.softmax(logits, axis=-1), axis=-1))

        evaluation = {
            'perplexity': perplexity,
            'confidence': confidence,
        }

        return generated_ids, evaluation

    def calculate_coherence(self, text: str) -> float:
        # This is a very simple coherence check. In practice, you'd want a more sophisticated method.
        words = text.split()
        if len(words) < 2:
            return 1.0
        
        coherence = 0
        for i in range(len(words) - 1):
            # Check if consecutive words often appear together in the training data
            # This would require access to training data statistics, which we don't have here
            # So we'll use a placeholder value
            coherence += 0.5  # placeholder
        
        return coherence / (len(words) - 1)

    def self_evaluate(self, generated_text: str, evaluation_metrics: Dict) -> Dict:
        coherence = self.calculate_coherence(generated_text)
        evaluation_metrics['coherence'] = coherence

        # Interpret the metrics
        if evaluation_metrics['perplexity'] < 10 and evaluation_metrics['confidence'] > 0.8 and coherence > 0.7:
            evaluation_metrics['overall_quality'] = 'High'
        elif evaluation_metrics['perplexity'] < 50 and evaluation_metrics['confidence'] > 0.6 and coherence > 0.5:
            evaluation_metrics['overall_quality'] = 'Medium'
        else:
            evaluation_metrics['overall_quality'] = 'Low'

        return evaluation_metrics
