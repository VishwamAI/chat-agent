import jax
import jax.numpy as jnp
from transformers import FlaxBertForSequenceClassification, AutoTokenizer
from transformers import PreTrainedTokenizer
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
    split_index = x.shape[-1] // 2
    x1, x2 = jnp.split(x, [split_index], axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

# Define the ImprovedAttention class
class ImprovedAttention(nn.Module):
    config: Dict

    def setup(self):
        self.num_heads = self.config['num_heads']
        self.head_dim = self.config['head_dim']
        self.qkv_dense = nn.Dense(3 * self.num_heads * self.head_dim)
        print(f"Setup - num_heads: {self.num_heads}, head_dim: {self.head_dim}, qkv_dense output dim: {3 * self.num_heads * self.head_dim}")

    @property
    def rotary_emb(self):
        return self._rotary_emb

    def _create_rotary_emb(self, seq_len, num_heads):
        # Create rotary positional embeddings dynamically
        head_dim = self.head_dim
        sin = jnp.sin(jnp.arange(seq_len * head_dim)).reshape((1, seq_len, head_dim))
        cos = jnp.cos(jnp.arange(seq_len * head_dim)).reshape((1, seq_len, head_dim))
        sin = jnp.broadcast_to(sin, (1, seq_len, num_heads, head_dim))
        cos = jnp.broadcast_to(cos, (1, seq_len, num_heads, head_dim))
        print(f"Generated sin shape: {sin.shape}")
        print(f"Generated cos shape: {cos.shape}")
        return sin, cos

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[jnp.ndarray] = None):
        if len(x.shape) == 2:
            x = x[:, :, None]  # Add a third dimension if x is two-dimensional
        print(f"Input tensor shape before unpacking: {x.shape}")

        # Check if x has three dimensions and reshape accordingly
        if len(x.shape) == 3:
            batch_size, seq_len, embed_dim = x.shape
            num_heads = self.num_heads
            head_dim = self.head_dim
            print(f"Configuration values - num_heads: {num_heads}, head_dim: {head_dim}, embed_dim: {embed_dim}")
            if head_dim == 0 or num_heads == 0:
                print(f"Invalid head_dim or num_heads: head_dim={head_dim}, num_heads={num_heads}")
                raise ValueError(f"Invalid head_dim or num_heads: head_dim={head_dim}, num_heads={num_heads}")
            # Ensure embed_dim matches the expected value
            expected_embed_dim = num_heads * head_dim
            if embed_dim != expected_embed_dim:
                print(f"Embedding dimension mismatch: expected {expected_embed_dim}, but got {embed_dim}")
                raise ValueError(f"Embedding dimension mismatch: expected {expected_embed_dim}, but got {embed_dim}")
            print(f"Input tensor shape before reshaping: {x.shape}")
            x = x.reshape(batch_size, seq_len, num_heads, head_dim)  # Reshape to match the expected shape
            print(f"Input tensor shape after reshaping: {x.shape}")
        else:
            raise ValueError(f"Input tensor must have 2 or 3 dimensions, but got {len(x.shape)} dimensions")

        print(f"Input tensor shape after unpacking: {x.shape}")

        # Log the shape of x before applying qkv_dense
        print(f"Input tensor shape before qkv_dense: {x.shape}")
        print(f"Input tensor values before qkv_dense: {x}")

        # Apply qkv_dense to the reshaped tensor
        qkv = self.qkv_dense(x)  # Pass the reshaped tensor to qkv_dense

        # Log the shape and values of qkv after applying qkv_dense
        print(f"qkv shape after qkv_dense: {qkv.shape}")
        print(f"qkv values after qkv_dense: {qkv}")

        # Ensure qkv has the expected shape
        expected_qkv_shape = (batch_size, seq_len, 3 * self.num_heads * self.head_dim)
        print(f"Expected qkv shape: {expected_qkv_shape}")
        assert qkv.shape == expected_qkv_shape, f"Expected qkv shape {expected_qkv_shape}, but got {qkv.shape}"

        # Reshape qkv to match the expected shape and split into q, k, v
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)  # Reshape to match the expected shape
        print(f"qkv shape after reshaping: {qkv.shape}")
        q, k, v = jnp.split(qkv, 3, axis=2)  # Split along the third axis to ensure correct shapes

        # Log the shapes of qkv, q, k, and v
        print(f"qkv shape: {qkv.shape}")
        print(f"q shape after splitting: {q.shape}")
        print(f"k shape after splitting: {k.shape}")
        print(f"v shape after splitting: {v.shape}")

        sincos = self._create_rotary_emb(seq_len, self.num_heads)

        # Log the shapes before applying rotary positional embeddings
        print(f"q shape before apply_rotary_pos_emb: {q.shape}")
        print(f"k shape before apply_rotary_pos_emb: {k.shape}")
        print(f"sin shape before apply_rotary_pos_emb: {sincos[0].shape}")
        print(f"cos shape before apply_rotary_pos_emb: {sincos[1].shape}")

        # Ensure q and k have the correct shape before applying rotary positional embeddings
        expected_embed_dim = self.num_heads * self.head_dim
        if q.shape[-1] != expected_embed_dim or k.shape[-1] != expected_embed_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {expected_embed_dim}, but got q shape {q.shape[-1]} and k shape {k.shape[-1]}")

        # Additional debug logging to track tensor shapes
        print(f"q shape before apply_rotary_pos_emb: {q.shape}")
        print(f"k shape before apply_rotary_pos_emb: {k.shape}")

        # Print the shapes of q and k before calling apply_rotary_pos_emb
        print(f"q shape before apply_rotary_pos_emb: {q.shape}")
        print(f"k shape before apply_rotary_pos_emb: {k.shape}")

        q = apply_rotary_pos_emb(q, sincos, self.head_dim, self.num_heads)
        k = apply_rotary_pos_emb(k, sincos, self.head_dim, self.num_heads)

        # Log the shapes after applying rotary positional embeddings
        print(f"q shape after apply_rotary_pos_emb: {q.shape}")
        print(f"k shape after apply_rotary_pos_emb: {k.shape}")

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
            print(f"Mask shape before broadcasting: {mask.shape}")
            print(f"Attention tensor shape: {attn.shape}")
            mask = mask[:, :, :attn.shape[-2], :attn.shape[-1]]  # Slice mask to match attention tensor's dimensions
            mask = jnp.broadcast_to(mask, (batch_size, self.num_heads, attn.shape[-2], attn.shape[-1]))  # Ensure mask is expanded to match attn tensor's shape
            print(f"Mask shape after broadcasting: {mask.shape}")
            assert mask.shape == attn.shape, f"Mask shape {mask.shape} does not match attention tensor shape {attn.shape}"
            attn = jnp.where(mask, attn, float('-inf'))

        attn_weights = jax.nn.softmax(attn, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)

        return attn_output

def apply_rotary_pos_emb(x, sincos, head_dim, num_heads):
    sin, cos = sincos
    print(f"x shape: {x.shape}")
    print(f"head_dim: {head_dim}")
    print(f"num_heads: {num_heads}")
    print(f"sin shape before split: {sin.shape}")
    print(f"cos shape before split: {cos.shape}")

    # Ensure x has the correct shape before the split
    expected_embed_dim = num_heads * head_dim
    if x.shape[-1] != expected_embed_dim:
        raise ValueError(f"Embedding dimension mismatch: expected {expected_embed_dim}, but got {x.shape[-1]}")

    # Directly reshape x to the desired shape
    print(f"x shape before reshaping: {x.shape}")
    x = x.reshape((x.shape[0], x.shape[1], num_heads, head_dim))
    print(f"x shape after reshaping: {x.shape}")

    # Reshape sin and cos to match the dimensions of x for broadcasting
    print(f"sin shape before reshaping: {sin.shape}")
    print(f"cos shape before reshaping: {cos.shape}")
    sin = sin.reshape((1, x.shape[1], num_heads, head_dim))
    cos = cos.reshape((1, x.shape[1], num_heads, head_dim))
    print(f"sin shape after reshaping: {sin.shape}")
    print(f"cos shape after reshaping: {cos.shape}")

    # Ensure x has the correct shape before rotation
    if len(x.shape) != 4:
        x = x.reshape((x.shape[0], x.shape[1], num_heads, head_dim))
    if x.shape[-1] != head_dim:
        raise ValueError(f"Shape mismatch: x last dimension {x.shape[-1]} does not match head_dim {head_dim}")
    print(f"x shape after reshaping: {x.shape}")

    x_rotated = (x * cos) + (rotate_half(x) * sin)
    print(f"x_rotated shape after element-wise operations: {x_rotated.shape}")
    assert x_rotated.shape == x.shape, f"Shape mismatch: x_rotated shape {x_rotated.shape}, x shape {x.shape}"
    return x_rotated



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
        # Log the shape of x before layer normalization
        print(f"x shape before layer_norm1: {x.shape}")

        x = self.layer_norm1(x)

        # Log the shape of x after layer normalization
        print(f"x shape after layer_norm1: {x.shape}")

        attention_output = self.attention(x, mask, kv_cache)

        # Log the shape of attention_output after attention mechanism
        print(f"attention_output shape: {attention_output.shape}")

        attention_output = self.dropout(attention_output, deterministic=not is_training)

        # Log the shape of attention_output after dropout
        print(f"attention_output shape after dropout: {attention_output.shape}")

        x = x + attention_output

        # Log the shape of x after adding attention_output
        print(f"x shape after adding attention_output: {x.shape}")

        x = self.layer_norm2(x)

        # Log the shape of x after layer normalization
        print(f"x shape after layer_norm2: {x.shape}")

        ff_output = self.feed_forward(x)

        # Log the shape of ff_output after feed-forward network
        print(f"ff_output shape: {ff_output.shape}")

        ff_output = self.dropout(ff_output, deterministic=not is_training)

        # Log the shape of ff_output after dropout
        print(f"ff_output shape after dropout: {ff_output.shape}")

        x = x + ff_output

        # Log the shape of x after adding ff_output
        print(f"x shape after adding ff_output: {x.shape}")

        return x

class ImprovedVishwamAIModel(nn.Module):
    config: Dict
    tokenizer: PreTrainedTokenizer
    bert_model: nn.Module
    num_layers: int
    vocab_size: int

    def setup(self):
        self.transformer_blocks = [ImprovedTransformerBlock(self.config) for _ in range(self.num_layers)]

    def _create_mask(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        return (input_ids != self.tokenizer.pad_token_id).astype(jnp.float32)

    def __call__(self, inputs: jnp.ndarray, is_training: bool = False, kv_cache: Optional[Dict] = None) -> jnp.ndarray:
        # Log the shape of inputs before reshaping
        print(f"inputs shape before reshaping: {inputs.shape}")

        input_ids = inputs.reshape(-1, inputs.shape[-1])

        # Log the shape of input_ids after reshaping
        print(f"input_ids shape after reshaping: {input_ids.shape}")

        attention_mask = (input_ids != self.tokenizer.pad_token_id).astype(jnp.float32)

        # Log the shape of attention_mask
        print(f"attention_mask shape: {attention_mask.shape}")

        if not hasattr(input_ids, 'shape'):
            raise TypeError("input_ids is not a valid tensor with the shape attribute")

        input_ids = jax.device_put(input_ids)

        bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_outputs.logits

        # Log the shape of x after BERT model
        print(f"x shape after BERT model: {x.shape}")

        mask = self._create_mask(input_ids)

        # Log the shape of mask
        print(f"mask shape: {mask.shape}")

        if kv_cache is None:
            kv_cache = [{'k': None, 'v': None} for _ in range(self.num_layers)]

        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x, mask, kv_cache[i], is_training)

            # Log the shape of x after each transformer block
            print(f"x shape after transformer block {i}: {x.shape}")

        dense_layer = nn.Dense(self.vocab_size)
        return dense_layer(x), kv_cache

class ImprovedVishwamAIModel(nn.Module):
    config: Dict

    def setup(self):
        self.embed_dim = self.config['embed_dim']
        self.num_layers = self.config['num_layers']
        self.vocab_size = self.config['vocab_size']
        self.head_dim = self.config['head_dim']  # Use head_dim from the configuration
        self.num_heads = self.config['num_heads']
        logger.debug(f"Model setup - embed_dim: {self.embed_dim}, num_layers: {self.num_layers}, vocab_size: {self.vocab_size}, head_dim: {self.head_dim}, num_heads: {self.num_heads}")

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
