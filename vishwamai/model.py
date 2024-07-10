"""Inference-only Vishwamai model implementation."""

# Copyright 2024 Vishwamai Org
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import gc
import os
import re
import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Any, List, Optional, Sequence, Tuple, Union

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

from vishwamai import config as vishwamai_config
from vishwamai import tokenizer

# Removed unnecessary imports from grok-1

from lark import Lark

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Obtain the rotated counterpart of each feature"""
    x1, x2 = torch.split(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class Sampler(nn.Module):

    def __init__(self, vocab_size: int, config: vishwamai_config.VishwamaiConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        hidden_states = hidden_states.index_select(
            1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1],
                                   device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True).squeeze(dim=-1)
        return next_token_ids, logits



class RotaryEmbedding(nn.Module):
    """Applies rotary embeddings (RoPE) to the input sequence tensor,
    as described in https://arxiv.org/abs/2104.09864.

    Attributes:
        dim (int): Dimensionality of the feature vectors
        base_exponent (int): Base exponent to compute embeddings from
    """

    def __init__(
        self,
        dim: int,
        base_exponent: int = 10000,
    ):
        super().__init__()
        self.dim = dim
        self.base_exponent = base_exponent
        assert self.dim % 2 == 0

    def forward(
        self,
        x: torch.Tensor,
        seq_dim: int,
        offset: torch.Tensor,
        const_position: Optional[int] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fprop_dtype = x.dtype
        exponents = torch.arange(0, self.dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.base_exponent ** (exponents / self.dim))

        if offset.shape == ():
            offset = offset.unsqueeze(0)

        if const_position:
            t = const_position * torch.ones(
                (1, x.shape[seq_dim]),
                dtype=torch.float32,
            )
        elif t is None:
            t = torch.arange(x.shape[seq_dim], dtype=torch.float32) + offset.unsqueeze(-1)
        phase = torch.einsum("bi,j->bij", t, inv_freq)
        phase = torch.tile(phase, (1, 2))[:, :, None, :]

        x = x * torch.cos(phase) + rotate_half(x) * torch.sin(phase)
        x = x.to(fprop_dtype)

        return x


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        self.quant = quant
        if quant:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = ColumnParallelLinear(
                in_features, out_features, bias=False, gather_output=False
            )

    def forward(self, x):
        if self.quant:
            weight = self.weight * self.weight_scaler.unsqueeze(-1)
            output = F.linear(x, weight)
        else:
            output = self.weight(x)
        return output


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        self.quant = quant
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = VocabParallelEmbedding(
                num_embeddings, embedding_dim, gather_output=False
            )

    def forward(self, x):
        if self.quant:
            weight = self.weight * self.weight_scaler.unsqueeze(-1)
            output = F.embedding(x, weight)
        else:
            output = self.weight(x)
        return output


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)



class VishwamaiMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant: bool,
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True
        )

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class VishwamaiAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        attn_logit_softcapping: Optional[float],
        query_pre_attn_scalar: Optional[int],
        head_dim: int,
        quant: bool,
        attn_type: vishwamai_config.AttentionType,
        sliding_window_size: Optional[int] = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.qkv_proj = ColumnParallelLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
        )

        self.attn_type = attn_type
        self.sliding_window_size = sliding_window_size
        self.attn_logit_softcapping = attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
                               dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding.
        rotary_embedding = RotaryEmbedding(dim=self.head_dim)
        xq = rotary_embedding(xq, seq_dim=1, offset=kv_write_indices)
        xk = rotary_embedding(xk, seq_dim=1, offset=kv_write_indices)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        q.mul_(self.scaling)
        scores = torch.matmul(q, k.transpose(2, 3))
        if (
            self.attn_type == vishwamai_config.AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
        ):
            all_ones = torch.ones_like(mask)
            sliding_mask = torch.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * torch.tril(all_ones, self.sliding_window_size - 1)
            mask = torch.where(sliding_mask == 1, mask, -2.3819763e38)
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.o_proj(output)
        return output


class VishwamaiDecoderLayer(nn.Module):

    def __init__(
        self,
        config: vishwamai_config.VishwamaiConfig,
    ):
        super().__init__()
        self.self_attn = VishwamaiAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            head_dim=config.head_dim,
            quant=config.quant,
            attn_type=vishwamai_config.AttentionType.GLOBAL,
        )
        self.mlp = VishwamaiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Vishwamai2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: vishwamai_config.VishwamaiConfig,
        attn_type: vishwamai_config.AttentionType,
    ):
        super().__init__()
        self.self_attn = VishwamaiAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            query_pre_attn_scalar=config.query_pre_attn_scalar,
            head_dim=config.head_dim,
            quant=config.quant,
            attn_type=attn_type,
            sliding_window_size=config.sliding_window_size,
        )
        self.mlp = VishwamaiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MathReasoningModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Define neural network components for math reasoning
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask=None):
        # Implement forward pass for math reasoning
        output = self.linear1(input_ids)
        output = self.activation(output)
        output = self.linear2(output)
        return output

class VishwamaiModel(nn.Module):
    def __init__(self, config: vishwamai_config.VishwamaiConfig):
        super().__init__()
        self.config = config
        self.math_reasoning_module = MathReasoningModule(config)
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if config.architecture == vishwamai_config.Architecture.VISHWAMAI_1:
                self.layers.append(VishwamaiDecoderLayer(config))
            elif config.architecture == vishwamai_config.Architecture.VISHWAMAI_2:
                attn_type = (
                    config.attn_types[i]
                    if config.attn_types is not None
                    else vishwamai_config.AttentionType.GLOBAL
                )
                self.layers.append(Vishwamai2DecoderLayer(config, attn_type))
            else:
                raise ValueError(f'Unknown architecture: {config.architecture}')
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        math_reasoning_output = self.math_reasoning_module(hidden_states)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class VishwamaiForCausalLM(nn.Module):

    def __init__(
        self,
        config: vishwamai_config.VishwamaiConfig,
    ):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.sampler = Sampler(vocab_size, config)

        # Removed pre-compute rotary embedding table and freqs_cis buffer

        # Initialize grok-1 components with error handling
        # Removed grok-1 components initialization

        # Define an expanded grammar for the Earley parser
        grammar = """
            start: sentence

            sentence: noun_phrase verb_phrase
                    | noun_phrase verb_phrase conjunction sentence

            noun_phrase: article noun
                       | article adjective noun

            verb_phrase: verb noun_phrase
                       | verb adverb noun_phrase

            article: "a" | "the"
            noun: "cat" | "dog" | "man" | "woman"
            verb: "sees" | "likes" | "chases" | "finds"
            adjective: "big" | "small" | "quick" | "lazy"
            adverb: "quickly" | "slowly"
            conjunction: "and" | "but"
        """

        # Initialize the Earley parser with the expanded grammar
        self.earley_parser = Lark(grammar, start='start', ambiguity='explicit')

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use T5 model for the forward pass
        outputs = self.model(
            input_ids=input_token_ids,
            attention_mask=mask,
            decoder_input_ids=output_positions,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state

        embedder_weight = self.model.shared.weight
        if self.config.quant:
            embedder_weight = (
                embedder_weight * self.model.shared.weight_scaler.unsqueeze(-1)
            )

        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    def sample_top_p(self, probs, p):
        """
        Perform top-p (nucleus) sampling on a probability distribution.

        Args:
            probs (torch.Tensor): Probability distribution tensor.
            p (float): Probability threshold for top-p sampling.

        Returns:
            torch.Tensor: Sampled token indices.

        Note:
            Top-p sampling selects the smallest set of tokens whose cumulative probability mass
            exceeds the threshold p. The distribution is renormalized based on the selected tokens.
        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: float = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
        num_beams: int = 5,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        use_nucleus_sampling: bool = True,
        use_beam_search: bool = True,
        length_penalty: float = 1.0,
        coverage_penalty: float = 0.0,
        use_top_k_sampling: bool = False,
        use_greedy_decoding: bool = False,
        constraints: Optional[List[str]] = None,  # New parameter for constrained decoding
    ) -> Union[str, Sequence[str]]:
        """
        Generates responses for given prompts using Vishwamai model.

        Args:
            prompts (Union[str, Sequence[str]]): Input prompts for text generation.
            device (Any): Device to run the model on (e.g., 'cpu' or 'cuda').
            output_len (int, optional): Length of the generated output. Defaults to 100.
            temperature (float, optional): Sampling temperature. Defaults to 0.95.
            top_p (float, optional): Probability threshold for top-p (nucleus) sampling. Defaults to 1.0.
            top_k (int, optional): Number of top tokens to consider for top-k sampling. Defaults to 100.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            repetition_penalty (float, optional): Penalty for repeated tokens. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): Size of n-grams that should not be repeated. Defaults to 0.
            use_nucleus_sampling (bool, optional): Whether to use nucleus sampling. Defaults to True.
            use_beam_search (bool, optional): Whether to use beam search. Defaults to True.
            length_penalty (float, optional): Penalty for sequence length. Defaults to 1.0.
            coverage_penalty (float, optional): Penalty for coverage. Defaults to 0.0.
            use_top_k_sampling (bool, optional): Whether to use top-k sampling. Defaults to False.
            use_greedy_decoding (bool, optional): Whether to use greedy decoding. Defaults to False.
            constraints (Optional[List[str]], optional): List of constraints for constrained decoding. Defaults to None.

        Returns:
            Union[str, Sequence[str]]: Generated responses.
        """
        # Input validation
        if not (0.0 < temperature <= 1.0):
            raise ValueError("Temperature must be between 0.0 and 1.0")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("Top-p must be between 0.0 and 1.0")
        if not (use_nucleus_sampling or use_beam_search or use_top_k_sampling or use_greedy_decoding):
            raise ValueError("At least one of use_nucleus_sampling, use_beam_search, use_top_k_sampling, or use_greedy_decoding must be True")

        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        # Tokenize prompts
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate text using T5 model
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=output_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=True,
        )

        # Decode generated text
        results = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # If a string was provided as input, return a string as output.
        return results[0] if is_str_prompt else results

    def bi_directional_generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: float = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
        use_nucleus_sampling: bool = True,
        use_beam_search: bool = True,
        use_top_k_sampling: bool = False,
        use_greedy_decoding: bool = False,
    ) -> Union[str, Sequence[str]]:
        """
        Generates responses for given prompts using Vishwamai model in both forward and backward directions.

        Args:
            prompts (Union[str, Sequence[str]]): Input prompts for text generation.
            device (Any): Device to run the model on (e.g., 'cpu' or 'cuda').
            output_len (int, optional): Length of the generated output. Defaults to 100.
            temperature (float, optional): Sampling temperature. Defaults to 0.95.
            top_p (float, optional): Probability threshold for top-p (nucleus) sampling. Defaults to 1.0.
            top_k (int, optional): Number of top tokens to consider for top-k sampling. Defaults to 100.
            use_nucleus_sampling (bool, optional): Whether to use nucleus sampling. Defaults to True.
            use_beam_search (bool, optional): Whether to use beam search. Defaults to True.
            use_top_k_sampling (bool, optional): Whether to use top-k sampling. Defaults to False.
            use_greedy_decoding (bool, optional): Whether to use greedy decoding. Defaults to False.

        Returns:
            Union[str, Sequence[str]]: Generated responses.
        """
        # Forward generation
        forward_results = self.generate(
            prompts, device, output_len, temperature, top_p, top_k,
            use_nucleus_sampling=use_nucleus_sampling,
            use_beam_search=use_beam_search,
            use_top_k_sampling=use_top_k_sampling,
            use_greedy_decoding=use_greedy_decoding
        )

        # Reverse prompts for backward generation
        reversed_prompts = [prompt[::-1] for prompt in prompts]
        backward_results = self.generate(
            reversed_prompts, device, output_len, temperature, top_p, top_k,
            use_nucleus_sampling=use_nucleus_sampling,
            use_beam_search=use_beam_search,
            use_top_k_sampling=use_top_k_sampling,
            use_greedy_decoding=use_greedy_decoding
        )

        # Combine forward and backward results
        combined_results = [f + b[::-1] for f, b in zip(forward_results, backward_results)]
        return combined_results

    def load_weights(self, model_path: str):
        if os.path.isfile(model_path):
            self.load_state_dict(
                torch.load(
                    model_path, mmap=True, weights_only=True,
                )['model_state_dict'],
                strict=False,
            )
        else:
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                self.load_state_dict(state_dict, strict=False)
                del state_dict  # Save memory.
                gc.collect()

    def calculate_metrics(self, generated_texts: List[str], reference_texts: List[str]) -> dict:
        """
        Calculate evaluation metrics for generated texts.

        Args:
            generated_texts (List[str]): List of generated text sequences.
            reference_texts (List[str]): List of reference text sequences.

        Returns:
            dict: Dictionary containing evaluation metrics (e.g., BLEU, ROUGE, METEOR, CIDEr).
        """
        from datasets import load_metric

        # Load BLEU, ROUGE, METEOR, and CIDEr metrics
        bleu_metric = load_metric("bleu")
        rouge_metric = load_metric("rouge")
        meteor_metric = load_metric("meteor")
        cider_metric = load_metric("cider")

        # Compute BLEU score
        bleu_score = bleu_metric.compute(predictions=generated_texts, references=reference_texts)

        # Compute ROUGE score
        rouge_score = rouge_metric.compute(predictions=generated_texts, references=reference_texts)

        # Compute METEOR score
        meteor_score = meteor_metric.compute(predictions=generated_texts, references=reference_texts)

        # Compute CIDEr score
        cider_score = cider_metric.compute(predictions=generated_texts, references=reference_texts)

        return {
            "BLEU": bleu_score,
            "ROUGE": rouge_score,
            "METEOR": meteor_score,
            "CIDEr": cider_score,
        }

    def validate_tokens_with_constraints(self, next_token_ids, current_sequence, constraints):
        valid_tokens = []
        for token in next_token_ids:
            sequence = current_sequence + [token]
            sequence_str = ' '.join([self.tokenizer.decode([t]) for t in sequence])
            try:
                self.earley_parser.parse(sequence_str)
                if constraints:
                    if all(constraint in sequence_str for constraint in constraints):
                        valid_tokens.append(token)
                else:
                    valid_tokens.append(token)
            except:
                continue
        if not valid_tokens:
            valid_tokens = [self.tokenizer.pad_id] * len(next_token_ids)
        return valid_tokens
