import jax
import jax.numpy as jnp
from transformers import FlaxBertForSequenceClassification, AutoTokenizer
from typing import Dict, Optional, Tuple, List
from functools import partial
import sympy as sp
import optax
import logging
import flax.linen as nn
from flax.training import train_state
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 't')

def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

# Define the ImprovedAttention class
class ImprovedAttention(nn.Module):
    config: Dict

    def setup(self):
        self.num_heads = self.config['num_heads']
        self.head_dim = self.config['head_dim']
        self.qkv_dense = nn.Dense(3 * self.num_heads * self.head_dim)

    @property
    def rotary_emb(self):
        return self._rotary_emb

    def _create_rotary_emb(self, seq_len, num_heads):
        # Create rotary positional embeddings dynamically
        head_dim = self.head_dim
        sin = jnp.sin(jnp.arange(seq_len * head_dim)).reshape((1, seq_len, 1, head_dim))
        cos = jnp.cos(jnp.arange(seq_len * head_dim)).reshape((1, seq_len, 1, head_dim))
        sin = jnp.broadcast_to(sin, (1, seq_len, num_heads, head_dim))
        cos = jnp.broadcast_to(cos, (1, seq_len, num_heads, head_dim))
        logger.debug(f"Generated sin shape: {sin.shape}")
        logger.debug(f"Generated cos shape: {cos.shape}")
        return sin, cos

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[jnp.ndarray] = None):
        if len(x.shape) == 2:
            x = x[:, :, None]  # Add a third dimension if x is two-dimensional
        batch_size, seq_len, embed_dim = x.shape
        logger.debug(f"Input tensor shape: {x.shape}")

        # Ensure x has the correct shape
        expected_embed_dim = self.num_heads * self.head_dim
        if embed_dim != expected_embed_dim:
            if embed_dim == self.head_dim:
                x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            elif embed_dim == 1:
                x = jnp.tile(x, (1, 1, expected_embed_dim))
                x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            else:
                # Handle cases where embed_dim is not equal to head_dim or 1
                x = x.reshape(batch_size, seq_len, embed_dim // self.head_dim, self.head_dim)
                if x.shape[2] != self.num_heads:
                    raise ValueError(f"Number of heads mismatch: expected {self.num_heads}, but got {x.shape[2]}")
            if x.shape[-1] != self.head_dim:
                raise ValueError(f"Head dimension mismatch: expected {self.head_dim}, but got {x.shape[-1]}")
        assert x.shape == (batch_size, seq_len, self.num_heads, self.head_dim), f"Embedding dimension must match num_heads * head_dim, but got {x.shape} instead of {(batch_size, seq_len, self.num_heads, self.head_dim)}"
        logger.debug(f"Reshaped input tensor shape: {x.shape}")

        qkv = self.qkv_dense(x.reshape(batch_size, seq_len, -1))  # Flatten the last two dimensions before passing to qkv_dense
        logger.debug(f"qkv shape after qkv_dense: {qkv.shape}")
        expected_qkv_shape = (batch_size, seq_len, 3 * self.num_heads * self.head_dim)
        assert qkv.shape == expected_qkv_shape, f"Expected qkv shape {expected_qkv_shape}, but got {qkv.shape}"
        qkv = qkv.reshape(batch_size, seq_len, 3 * self.num_heads * self.head_dim)  # Reshape to match the expected shape
        q, k, v = jnp.split(qkv, 3, axis=-1)  # Split along the last axis to ensure correct shapes

        # Log the shapes of qkv, q, k, and v
        logger.debug(f"qkv shape: {qkv.shape}")
        logger.debug(f"q shape after splitting: {q.shape}")
        logger.debug(f"k shape after splitting: {k.shape}")
        logger.debug(f"v shape after splitting: {v.shape}")

        sincos = self._create_rotary_emb(seq_len, self.num_heads)

        # Log the shapes before applying rotary positional embeddings
        logger.debug(f"q shape before apply_rotary_pos_emb: {q.shape}")
        logger.debug(f"k shape before apply_rotary_pos_emb: {k.shape}")
        logger.debug(f"sin shape before apply_rotary_pos_emb: {sincos[0].shape}")
        logger.debug(f"cos shape before apply_rotary_pos_emb: {sincos[1].shape}")

        q = apply_rotary_pos_emb(q, sincos, self.head_dim)
        k = apply_rotary_pos_emb(k, sincos, self.head_dim)

        # Log the shapes after applying rotary positional embeddings
        logger.debug(f"q shape after apply_rotary_pos_emb: {q.shape}")
        logger.debug(f"k shape after apply_rotary_pos_emb: {k.shape}")

        if kv_cache is not None:
            if kv_cache['k'] is None:
                kv_cache['k'] = k
                kv_cache['v'] = v
            else:
                k = jnp.concatenate([kv_cache['k'], k], axis=1)
                v = jnp.concatenate([kv_cache['v'], v], axis=1)
                kv_cache['k'] = k
                kv_cache['v'] = v

        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)

        if mask is not None:
            logger.debug(f"Mask shape before broadcasting: {mask.shape}")
            logger.debug(f"Attention tensor shape: {attn.shape}")
            mask = mask[:, :, :attn.shape[-2], :attn.shape[-1]]  # Slice mask to match attention tensor's dimensions
            mask = jnp.broadcast_to(mask, (batch_size, self.num_heads, attn.shape[-2], attn.shape[-1]))  # Ensure mask is expanded to match attn tensor's shape
            logger.debug(f"Mask shape after broadcasting: {mask.shape}")
            assert mask.shape == attn.shape, f"Mask shape {mask.shape} does not match attention tensor shape {attn.shape}"
            attn = jnp.where(mask, attn, float('-inf'))

        attn_weights = jax.nn.softmax(attn, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)

        return attn_output

def apply_rotary_pos_emb(x, sincos, head_dim):
    sin, cos = sincos
    logger.debug(f"x shape: {x.shape}")
    if x.shape[-1] % (2 * head_dim) != 0:
        # Pad the last dimension of x to be a multiple of 2 * head_dim
        pad_size = (2 * head_dim) - (x.shape[-1] % (2 * head_dim))
        x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, pad_size)), mode='constant')
        logger.debug(f"Padded x shape: {x.shape}")
    x1, x2 = jnp.split(x, 2, axis=-1)
    logger.debug(f"x1 shape: {x1.shape}")
    logger.debug(f"x2 shape: {x2.shape}")
    logger.debug(f"sin shape: {sin.shape}")
    logger.debug(f"cos shape: {cos.shape}")

    # Reshape sin and cos to match the dimensions of x1 and x2 for broadcasting
    sin = sin.reshape((1, x1.shape[1], self.num_heads, head_dim))
    cos = cos.reshape((1, x1.shape[1], self.num_heads, head_dim))

    x_rotated = (x1 * cos) + (rotate_half(x1) * sin)
    logger.debug(f"x_rotated shape after reshaping: {x_rotated.shape}")
    logger.debug(f"x2 shape after reshaping: {x2.shape}")
    assert x_rotated.shape == x2.shape, f"Shape mismatch: x_rotated shape {x_rotated.shape}, x2 shape {x2.shape}"
    concatenated = jnp.concatenate([x_rotated, x2], axis=-1)
    logger.debug(f"concatenated shape: {concatenated.shape}")
    return concatenated



# Removed unnecessary debug logging statements and print statement used for debugging purposes

class MathReasoningLayer(nn.Module):
    config: Dict

    def setup(self):
        self.config = self.config

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def _tensor_to_expressions(self, x: jnp.ndarray) -> List[str]:
        expressions = []
        for val in x.flatten():
            expr = str(val)
            expressions.append(expr)
        return expressions

    def _batch_sympify(self, expressions: List[str], debug: bool = True) -> List[str]:
        return expressions

    def _expressions_to_tensor(self, expressions: List[str], shape: Tuple[int]) -> jnp.ndarray:
        # Convert list of string expressions back to tensor
        tensor_values = []
        for expr in expressions:
            try:
                value = float(expr)
            except ValueError:
                value = 0.0  # Handle invalid expressions gracefully
            tensor_values.append(value)
        return jnp.array(tensor_values).reshape(shape)

    def _generate_modular_problems(self, num_problems: int) -> List[str]:
        problems = []
        for _ in range(num_problems):
            # Generate a random mathematical expression
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            problem = f"Solve for x: {a}x + {b} = 0"
            problems.append(problem)
        return problems

    def _generate_compositional_problems(self, num_problems: int) -> List[str]:
        problems = []
        for _ in range(num_problems):
            # Generate a random mathematical expression
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            problem = f"Solve for x: {a}x + {b} = 0"
            problems.append(problem)
        return problems


class ImprovedTransformerBlock(nn.Module):
    config: Dict

    def setup(self):
        self.attention = ImprovedAttention(self.config)
        self.feed_forward = nn.Sequential([
            nn.Dense(self.config['ff_dim']),
            nn.gelu,
            nn.Dense(self.config['embed_dim']),
        ])
        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.config['dropout_rate'])

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[Dict] = None, is_training: bool = False) -> jnp.ndarray:
        x = self.layer_norm1(x)
        attention_output = self.attention(x, mask, kv_cache)
        attention_output = self.dropout(attention_output, deterministic=not is_training)

        # Log the shapes of x and attention_output before broadcasting
        logger.debug(f"x shape before broadcasting: {x.shape}")
        logger.debug(f"attention_output shape: {attention_output.shape}")

        # Ensure x and attention_output have compatible shapes
        if x.shape != attention_output.shape:
            if len(x.shape) == len(attention_output.shape):
                if x.size == attention_output.size:
                    x = jnp.reshape(x, attention_output.shape)  # Reshape x to match attention_output's shape
                else:
                    # Handle cases where the shapes are not directly compatible for reshaping
                    raise ValueError(f"Incompatible shapes for broadcasting: {x.shape} and {attention_output.shape}")
            else:
                # Handle cases where the number of dimensions differ
                while len(x.shape) < len(attention_output.shape):
                    x = jnp.expand_dims(x, axis=-1)  # Add new axes at the last dimension
                # Use jnp.tile to repeat x along the new axes to match the shape of attention_output
                tile_shape = [1] * len(x.shape)
                for i in range(len(x.shape)):
                    if x.shape[i] != attention_output.shape[i]:
                        tile_shape[i] = attention_output.shape[i] // x.shape[i]
                x = jnp.tile(x, tile_shape)
                logger.debug(f"x shape after tiling: {x.shape}")
                if x.shape != attention_output.shape:
                    raise ValueError(f"Incompatible number of dimensions for broadcasting: {x.shape} and {attention_output.shape}")

        # Log the shapes of x and attention_output after reshaping
        logger.debug(f"x shape after reshaping: {x.shape}")
        logger.debug(f"attention_output shape after reshaping: {attention_output.shape}")

        x = x + attention_output
        x = self.layer_norm2(x)
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output, deterministic=not is_training)

        # Log the shape of ff_output before ensuring compatibility with x
        logger.debug(f"ff_output shape before ensuring compatibility: {ff_output.shape}")

        # Ensure ff_output has the same shape as x before addition
        if ff_output.shape != x.shape:
            if ff_output.size == x.size:
                ff_output = jnp.reshape(ff_output, x.shape)  # Reshape ff_output to match x's shape
            else:
                # Slice ff_output to match the last dimension of x if possible
                if ff_output.shape[-1] > x.shape[-1]:
                    ff_output = ff_output[..., :x.shape[-1]]
                else:
                    raise ValueError(f"Incompatible shapes for broadcasting: {ff_output.shape} and {x.shape}")

        # Log the shape of ff_output after ensuring compatibility with x
        logger.debug(f"ff_output shape after ensuring compatibility: {ff_output.shape}")

        x = x + ff_output

        return x

class ImprovedVishwamAIModel(nn.Module):
    config: Dict

    def setup(self):
        self.embed_dim = self.config['embed_dim']
        self.num_layers = self.config['num_layers']
        self.vocab_size = self.config['vocab_size']
        self.head_dim = self.config['head_dim']  # Use head_dim from the configuration
        self.num_heads = self.config['num_heads']

        self.bert_model = FlaxBertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.transformer_blocks = [ImprovedTransformerBlock(self.config) for _ in range(self.num_layers)]

    def __call__(self, inputs: jnp.ndarray, is_training: bool = False, kv_cache: Optional[Dict] = None) -> jnp.ndarray:
        input_ids = inputs.reshape(-1, inputs.shape[-1])
        attention_mask = (input_ids != self.tokenizer.pad_token_id).astype(jnp.float32)

        if not hasattr(input_ids, 'shape'):
            raise TypeError("input_ids is not a valid tensor with the shape attribute")

        input_ids = jax.device_put(input_ids)

        bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_outputs.logits

        mask = self._create_mask(input_ids)

        if kv_cache is None:
            kv_cache = [{'k': None, 'v': None} for _ in range(self.num_layers)]

        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x, mask, kv_cache[i], is_training)

        dense_layer = nn.Dense(self.vocab_size)
        return dense_layer(x), kv_cache

    def _embed(self, x: jnp.ndarray) -> jnp.ndarray:
        embedding_matrix = hk.get_parameter("embedding_matrix",
                                            shape=[self.vocab_size, self.embed_dim],
                                            init=hk.initializers.TruncatedNormal(stddev=0.02))
        return jnp.take(embedding_matrix, x, axis=0)

    def _create_mask(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if self.config['pad_token_id'] is None:
            raise ValueError("pad_token_id is not set in the configuration.")
        mask = jnp.not_equal(inputs, self.config['pad_token_id']).astype(jnp.float32)
        mask = mask[:, None, None, :]
        seq_length = inputs.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), jnp.float32))
        mask = jnp.broadcast_to(mask, (inputs.shape[0], self.config['num_heads'], seq_length, seq_length))
        causal_mask = jnp.broadcast_to(causal_mask[None, None, :, :], (inputs.shape[0], self.config['num_heads'], seq_length, seq_length))
        mask = mask * causal_mask
        return mask

    def generate(self, input_ids: jnp.ndarray, max_length: int = 100, temperature: float = 1.0) -> jnp.ndarray:
        generated_ids = input_ids
        rng = jax.random.PRNGKey(0)
        for _ in range(max_length - input_ids.shape[1]):
            logits, _ = self(generated_ids, is_training=False)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = jax.random.categorical(rng, next_token_logits, axis=-1)
            generated_ids = jnp.concatenate([generated_ids, next_token[:, jnp.newaxis]], axis=-1)
        return generated_ids

    def generate_with_evaluation(self, input_ids: jnp.ndarray, kv_cache: Optional[Dict] = None, max_length: int = 100, temperature: float = 1.0) -> Tuple[jnp.ndarray, Dict]:
        generated_ids = input_ids
        total_log_probs = 0.0
        rng = jax.random.PRNGKey(0)
        for _ in range(max_length - input_ids.shape[1]):
            logits, kv_cache = self(generated_ids, is_training=False, kv_cache=kv_cache)
            next_token_logits = logits[:, -1, :] / temperature
            next_token_probs = jax.nn.softmax(next_token_logits, axis=-1)
            next_token = jax.random.categorical(rng, next_token_logits, axis=-1)
            generated_ids = jnp.concatenate([generated_ids, next_token[:, jnp.newaxis]], axis=-1)

class VishwamAILLM(nn.Module):
    config: Dict

    def setup(self):
        self.transformer = ImprovedVishwamAIModel(self.config)
        self.lm_head = nn.Dense(self.config['vocab_size'])

    def __call__(self, inputs: jnp.ndarray, is_training: bool = False, kv_cache: Optional[Dict] = None) -> Tuple[jnp.ndarray, Dict]:
        transformer_outputs, new_kv_cache = self.transformer(inputs, is_training, kv_cache)
        lm_logits = self.lm_head(transformer_outputs)
        return lm_logits, new_kv_cache

    def generate(self, input_ids: jnp.ndarray, max_length: int = 100, temperature: float = 1.0) -> jnp.ndarray:
        generated_ids = input_ids
        for _ in range(max_length - input_ids.shape[1]):
            logits, _ = self(generated_ids, is_training=False)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = jax.random.categorical(jax.random.PRNGKey(0), next_token_logits, axis=-1)
            generated_ids = jnp.concatenate([generated_ids, next_token[:, jnp.newaxis]], axis=-1)
        return generated_ids

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def generate_with_evaluation(self, input_ids: jnp.ndarray, kv_cache: Optional[Dict] = None, max_length: int = 100, temperature: float = 1.0) -> Tuple[jnp.ndarray, Dict]:
        generated_ids = input_ids
        total_log_probs = 0.0
        for _ in range(max_length - input_ids.shape[1]):
            logits, kv_cache = self(generated_ids, is_training=False, kv_cache=kv_cache)
            next_token_logits = logits[:, -1, :] / temperature
            next_token_probs = jax.nn.softmax(next_token_logits, axis=-1)
            next_token = jax.random.categorical(jax.random.PRNGKey(0), next_token_logits, axis=-1)
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
        words = text.split()
        if len(words) < 2:
            return 1.0

        coherence = 0
        for i in range(len(words) - 1):
            coherence += 0.5

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
