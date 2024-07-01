import jax
import jax.numpy as jnp
import haiku as hk
from transformers import BertModel, BertTokenizer
from typing import Dict, Optional, Tuple, List
from functools import partial
import sympy as sp
import optax
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def split_and_rotate(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    x1, x2 = jnp.split(x, 2, axis=-1)
    x_rotated = jnp.concatenate([-x2, x1], axis=-1)
    result = (x * cos) + (x_rotated * sin)
    del x1, x2, x_rotated, sin, cos  # Ensure intermediate variables are deleted
    jax.lax.create_token(result)  # Ensure result is materialized
    return result

class RotaryEmbedding(hk.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sin, self.cos = None, None

    def __call__(self, seq_len):
        import psutil
        memory_usage_before = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage before RotaryEmbedding calculations: {memory_usage_before:.2f} MiB")
        if self.sin is None or self.cos is None:
            inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim) / self.head_dim))
            t = jnp.arange(seq_len)
            freqs = jnp.outer(t, inv_freq)
            self.sin = jnp.sin(freqs).reshape(seq_len, self.head_dim).repeat(self.num_heads, axis=0).reshape(1, seq_len, self.num_heads, self.head_dim)
            self.cos = jnp.cos(freqs).reshape(seq_len, self.head_dim).repeat(self.num_heads, axis=0).reshape(1, seq_len, self.num_heads, self.head_dim)
            print(f"RotaryEmbedding - sin shape: {self.sin.shape}")
            print(f"RotaryEmbedding - cos shape: {self.cos.shape}")
        memory_usage_after = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage after RotaryEmbedding calculations: {memory_usage_after:.2f} MiB")
        return self.sin, self.cos

class ImprovedAttention(hk.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.num_heads = config['num_heads']
        self.head_dim = 32  # Adjust head_dim to 32 to match the actual dimensions
        self.rotary_emb = RotaryEmbedding(self.num_heads, self.head_dim)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[Dict] = None):
        seq_len = x.shape[1]
        qkv = hk.Linear(3 * self.num_heads * self.head_dim, with_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        import psutil
        memory_usage_before_reshape = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage before tensor reshaping: {memory_usage_before_reshape:.2f} MiB")
        q = q.reshape(-1, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(-1, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(-1, seq_len, self.num_heads, self.head_dim)
        memory_usage_after_reshape = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage after tensor reshaping: {memory_usage_after_reshape:.2f} MiB")

        sincos = self.rotary_emb(seq_len)
        memory_usage_before_q = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage before apply_rotary_pos_emb (q): {memory_usage_before_q:.2f} MiB")
        q = apply_rotary_pos_emb(q, sincos)
        memory_usage_after_q = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage after apply_rotary_pos_emb (q): {memory_usage_after_q:.2f} MiB")
        import gc
        gc.collect()

        memory_usage_before_k = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage before apply_rotary_pos_emb (k): {memory_usage_before_k:.2f} MiB")
        k = apply_rotary_pos_emb(k, sincos)
        memory_usage_after_k = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage after apply_rotary_pos_emb (k): {memory_usage_after_k:.2f} MiB")
        gc.collect()

        if kv_cache is not None:
            if kv_cache['k'] is None:
                kv_cache['k'] = k
                kv_cache['v'] = v
            else:
                k = jnp.concatenate([kv_cache['k'], k], axis=1)
                v = jnp.concatenate([kv_cache['v'], v], axis=1)
                kv_cache['k'] = k
                kv_cache['v'] = v

        memory_usage_before_matmul = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage before matrix multiplication: {memory_usage_before_matmul:.2f} MiB")
        print(f"Shape of q before matmul: {q.shape}")  # Debugging statement
        print(f"Shape of k before matmul: {k.shape}")  # Debugging statement
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        print(f"Shape of attn after matmul: {attn.shape}")  # Debugging statement
        attn = attn.reshape(q.shape[0], self.num_heads, seq_len, self.head_dim)  # Adjust shape to match mask tensor
        print(f"Shape of attn after reshaping: {attn.shape}")  # Debugging statement
        memory_usage_after_matmul = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage after matrix multiplication: {memory_usage_after_matmul:.2f} MiB")

        if mask is not None:
            print(f"Shape of mask before broadcasting: {mask.shape}")  # Debugging statement
            print(f"Shape of attn before broadcasting: {attn.shape}")  # Debugging statement
            mask = jnp.broadcast_to(mask, attn.shape)  # Ensure mask is expanded to match attn tensor's shape
            attn = jnp.where(mask, attn, float('-inf'))

        attn = jax.nn.softmax(attn, axis=-1)

        output = jnp.matmul(attn, v)
        return output.reshape(-1, seq_len, self.num_heads * self.head_dim)

class MathReasoningLayer(hk.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Directly return the input tensor without any modifications
        return x

    def _tensor_to_expressions(self, x: jnp.ndarray) -> List[str]:
        # Convert tensor to list of string expressions
        expressions = []
        for val in x.flatten():
            expr = str(val)
            # Add logic to handle mathematical symbols and expressions
            expressions.append(expr)
        return expressions

    def _batch_sympify(self, expressions: List[str], debug: bool = True) -> List[str]:
        # import time
        # import tracemalloc
        # from concurrent.futures import ThreadPoolExecutor

        # # Cache for storing previously computed results
        # cache = {}
        # solved_expressions = []

        # # Start memory profiling if debug is enabled
        # if debug:
        #     tracemalloc.start()

        # def sympify_expression(expr):
        #     if expr in cache:
        #         return cache[expr]
        #     else:
        #         solved_expr = sp.sympify(expr).evalf()
        #         cache[expr] = solved_expr
        #         return solved_expr

        # start_time = time.time()
        # with ThreadPoolExecutor() as executor:
        #     solved_expressions = list(executor.map(sympify_expression, expressions))
        # end_time = time.time()

        # print(f"Total sympify operation time: {end_time - start_time:.4f} seconds")

        # # Stop memory profiling and get memory usage if debug is enabled
        # if debug:
        #     current, peak = tracemalloc.get_traced_memory()
        #     tracemalloc.stop()
        #     print(f"Memory usage: Current = {current / 10**6:.2f} MB; Peak = {peak / 10**6:.2f} MB")

        # return solved_expressions
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

    # Generate a sequence of modular mathematical problems
    def _generate_modular_problems(self, num_problems: int) -> List[str]:
        problems = []
        for _ in range(num_problems):
            # Generate a random mathematical expression
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            problem = f"Solve for x: {a}x + {b} = 0"
            problems.append(problem)
        return problems

    # This will involve creating a sequence of mathematical problems where the output of one problem can serve as the input to another
    def _generate_compositional_problems(self, num_problems: int) -> List[str]:
        problems = []
        for _ in range(num_problems):
            # Generate a random mathematical expression
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            problem = f"Solve for x: {a}x + {b} = 0"
            problems.append(problem)
        return problems

class ImprovedTransformerBlock(hk.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.attention = ImprovedAttention(config)
        self.feed_forward = hk.Sequential([
            hk.Linear(config['ff_dim']),
            jax.nn.gelu,
            hk.Linear(config['embed_dim']),
        ])
        self.math_reasoning = MathReasoningLayer(config)
        self.layer_norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layer_norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.dropout = lambda x: hk.dropout(hk.next_rng_key(), config['dropout_rate'], x)
        self.optimizer = optax.adam(config['learning_rate'])
        rng_key = hk.next_rng_key()
        logger.debug(f"RNG key type: {type(rng_key)}")
        logger.debug(f"RNG key value: {rng_key}")
        self.opt_state = self.optimizer.init(rng_key)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[Dict] = None, is_training: bool = False) -> jnp.ndarray:
        import psutil
        memory_usage_before_attention = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage before attention: {memory_usage_before_attention:.2f} MiB")
        attention_output = self.attention(self.layer_norm1(x), mask, kv_cache)
        attention_output = self.dropout(attention_output) if is_training else attention_output
        x = x + attention_output
        memory_usage_after_attention = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage after attention: {memory_usage_after_attention:.2f} MiB")

        memory_usage_before_ff = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage before feed-forward: {memory_usage_before_ff:.2f} MiB")
        ff_output = self.feed_forward(self.layer_norm2(x))
        ff_output = self.dropout(ff_output) if is_training else ff_output
        x = x + ff_output
        memory_usage_after_ff = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MiB
        print(f"Memory usage after feed-forward: {memory_usage_after_ff:.2f} MiB")

        # Apply math reasoning layer
        math_output = self.math_reasoning(x)
        x = x + math_output

        return x

class ImprovedVishwamAIModel(hk.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.embed_dim = config['embed_dim']
        self.num_layers = config['num_layers']
        self.vocab_size = config['vocab_size']
        self.head_dim = 32  # Define head_dim as an attribute of the class
        self.num_heads = config['num_heads']  # Define num_heads as an attribute of the class
        self.config['head_dim'] = 32  # Add head_dim to the configuration

        # Instantiate BERT model and tokenizer
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __call__(self, inputs: jnp.ndarray, is_training: bool = False, kv_cache: Optional[Dict] = None) -> jnp.ndarray:
        # Ensure input_ids are correctly shaped as a 2D tensor
        input_ids = inputs.reshape(-1, inputs.shape[-1])

        # Create attention_mask directly from input_ids
        attention_mask = (input_ids != self.tokenizer.pad_token_id).astype(jnp.float32)

        # Check if input_ids is a valid tensor with the shape attribute
        if not hasattr(input_ids, 'shape'):
            raise TypeError("input_ids is not a valid tensor with the shape attribute")

        # Log the types and values of input_ids and attention_mask
        logger.debug(f"input_ids type: {type(input_ids)}, value: {input_ids}")
        logger.debug(f"attention_mask type: {type(attention_mask)}, value: {attention_mask}")

        # Convert input_ids to a JAX numpy array
        input_ids = jax.device_put(input_ids)

        # Convert input_ids to a PyTorch tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Pass inputs through BERT model
        bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_outputs.last_hidden_state

        mask = self._create_mask(input_ids)

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
        print(f"pad_token_id: {self.config['pad_token_id']}")  # Debugging statement
        if self.config['pad_token_id'] is None:
            raise ValueError("pad_token_id is not set in the configuration.")
        mask = jnp.not_equal(inputs, self.config['pad_token_id']).astype(jnp.float32)
        mask = mask[:, None, None, :]  # Expand mask dimensions to match attention tensor's shape
        seq_length = inputs.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), jnp.float32))
        mask = jnp.broadcast_to(mask, (mask.shape[0], self.num_heads, seq_length, seq_length))  # Adjust mask dimensions to match attention tensor's shape
        causal_mask = jnp.broadcast_to(causal_mask[None, None, :, :], (mask.shape[0], self.num_heads, seq_length, seq_length))  # Expand causal mask dimensions
        mask = mask * causal_mask  # Apply causal mask and adjust dimensions
        return mask

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

    def generate(self, input_ids: jnp.ndarray, max_length: int = 100, temperature: float = 1.0) -> jnp.ndarray:
        generated_ids = input_ids
        for _ in range(max_length - input_ids.shape[1]):
            logits, _ = self(generated_ids, is_training=False)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = jax.random.categorical(hk.next_rng_key(), next_token_logits, axis=-1)
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
