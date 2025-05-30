# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch LLaMA model."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_llama import LlamaConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"




global_inf_config = {
    
    # model setting
    'sys_length': None,
    'image_token_default_length': 576,
    'gen_token_idx': 0,

    # inference config
    'attention_weights': [None] * 32,
    'update_inference': False,
    'prefill_mask': None,
    'decode_mask': None,
    'prefill_position_ids': None,
    'decode_position_ids': None,
    'image_token_start': None,
    'image_token_current_length': None,
    'cur_text_num': None,
    'merge_text_start_token_idx': None,
    'merge_weights': [None] * 32,
    'compute_vision_index_list': [None] * 256,
    'uncompute_vision_index_list': [None] * 256,
    
    # vision adaptive attention
    'vision_adaptive_attention': False,
    'vision_adaptive_attention_rate': 0.5,
    'vision_adaptive_attention_compute_rate': 0.5,
    'vision_adaptive_attention_update_interval': -1,
    'use_customized_cuda': False,
    # 如果 vision_adaptive_attention_update_interval == -1, 则只初始过滤一下

    # merge
    'merge_attention': False,
    'merge_rate': 0.5,
    'merge_text_num': 30,
    'remove_text_token': False,
    'weaken_factor': 0.1,
    
    # prefill_filter_flag
    'prefill_filter_flag': False,
    'prefill_filter_layer': 2,
    'prefill_filter_rate': 0.2,

    'compare_full_tkn_del': False,


    'text_pruning_flag': False,
    'text_pruning_rate': 0.5,
    'text_pruning_interval': 1,
    'compare_text_direct_pruning': False,
    'text_init_cache': 1,

    'all_use_h2o': False,
    'h2o_hh_size': 100,
    'h2o_recent_size': 255,

    'text_use_h2o': False,
    'text_h20_hh_rate_in_cache': 0.5
}








class H2OKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache, hh_size=None, recent_size=None):

        if hh_size != None:

            self.hh_size = hh_size
            self.recent_size = recent_size
    
            self.cache_size = hh_size + recent_size

        self._update_hh_score(attn_score_cache)

        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)

        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape

        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        self.hh_score= self.hh_score[mask].view(num_heads, self.cache_size)

        return (k_hh_recent, v_hh_recent)

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape

        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size + num_coming]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(seq_len - self.recent_size + num_coming, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        self.hh_score= self.hh_score[mask].view(num_heads, self.cache_size)

        return (k_hh_recent, v_hh_recent)

    def _update_hh_score(self, attn_score_cache):

        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            attn_score_cache = attn_score_cache.sum(0).sum(1)
            attn_score_cache[:, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None




class TextPruneDirectKVCache:
    def __init__(
        self,
        cache_size=22,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache, cache_size):

        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0].size(self.k_seq_dim)

        self.cache_size = cache_size
        
        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape

        # num_new_tokens = attn_score_cache.shape[2]

        select_scores = attn_score_cache[0, :, -1, :]
        _, keep_topk = torch.topk(select_scores, self.cache_size, dim=-1)
        keep_idx = keep_topk.sort().values
        
        mask = torch.zeros(past_key_values[0].shape[1:3], dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)
        
        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)


        return (k_hh_recent, v_hh_recent)
    
    



class TextPruneHistoryKVCache:
    def __init__(
        self,
        pruning_rate=0.5,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        self.pruning_rate = pruning_rate
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache, cache_size):

        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0].size(self.k_seq_dim)

        self.cache_size = cache_size

        self._update_hh_score(attn_score_cache)
        
        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, text_length, head_dim = past_key_values[0].shape


        self.recent_size = int(self.cache_size / 2)
        self.hh_size = self.cache_size - self.recent_size



        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        weights_mask = torch.zeros((num_heads, text_length), dtype=torch.bool).to(past_key_values[0].device)
        weights_mask = weights_mask.scatter(-1, keep_idx, 1)
        
        
 
        # cur_text_weights = self.hh_score[:, :text_length]
        
        # text_remain_indices = cur_text_weights.topk(self.cache_size).indices
        # text_remain_indices = text_remain_indices.sort().values

        # weights_mask = torch.zeros_like(cur_text_weights, dtype=torch.bool, device=past_key_values[0].device)
        # weights_mask = weights_mask.scatter(-1, text_remain_indices, 1)

        k_hh_recent = past_key_values[0].squeeze()[weights_mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[weights_mask].view(bsz, num_heads, -1, head_dim)
        
    
        weights_mask = torch.zeros_like(self.hh_score, dtype=torch.bool, device=self.hh_score.device)
        weights_mask = weights_mask.scatter(-1, keep_idx, 1)
        weights_mask[:, text_length:] = True

        self.hh_score = self.hh_score[weights_mask].view(num_heads, -1)
        
        return (k_hh_recent, v_hh_recent)
    
    def _update_hh_score(self, attn_score_cache):

        num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:

            self.hh_score = attn_score_cache.new_zeros(attn_score_cache.shape[1], 32768)

            for text_idx in range(num_new_tokens):
                self.hh_score[:, text_idx] = attn_score_cache[0, :, text_idx, text_idx]

                if text_idx == 0:
                    continue

                percentile_num = torch.quantile(attn_score_cache[0, :, text_idx, :text_idx].float(), self.pruning_rate, dim=1)
                cumulative_weights = attn_score_cache[0, :, text_idx, :text_idx] - percentile_num.view(-1, 1)
                self.hh_score[:, :text_idx] += cumulative_weights
        
        else:

            assert num_new_tokens == 1

            self.hh_score[:, attn_score_cache.shape[-1]-1] = attn_score_cache[0, :, 0, -1]

            percentile_num = torch.quantile(attn_score_cache[0, :, 0, :].float(), self.pruning_rate, dim=1)
            cumulative_weights = attn_score_cache[0, :, 0, :-1] - percentile_num.view(-1, 1)

            self.hh_score[:, :cumulative_weights.shape[1]] += cumulative_weights


    def _clean_scores(self):
        self.hh_score = None
    
    




















def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @property
    def sin_cached(self):
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._cos_cached

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # In case static cache is used, it is an instance attribute.
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            if hasattr(self.config, "_pre_quantization_dtype"):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = layer.self_attn.o_proj.weight.dtype
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, device=device, dtype=dtype
            )

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # ==================================================

        if past_key_values is None:
            global_inf_config["gen_token_idx"] = 0

            global_inf_config["attention_weights"] = [None] * len(self.layers)

            image_indices = torch.where(self.init_image_token_position_mask[0])[0]
            global_inf_config["image_indices"] = image_indices
            global_inf_config["image_token_default_length"] = image_indices[-1].item() - image_indices[0].item() + 1

            text_indices = torch.where(~self.init_image_token_position_mask[0])[0]
            global_inf_config["text_indices"] = text_indices

            first_true_index = image_indices[0].item()
            global_inf_config["sys_length"] = first_true_index

            last_image_token = image_indices[-1].item()
            global_inf_config["text_start_idx"] = last_image_token + 1 

            global_inf_config["image_token_current_length"] = global_inf_config["image_token_default_length"]

            global_inf_config["decode_position_ids"] = None

        else:
            global_inf_config["gen_token_idx"] += 1      

        SYS_LENGTH = global_inf_config["sys_length"]

        if global_inf_config["vision_adaptive_attention_update_interval"] > 0:
            compute_part_vision_token = True
            if global_inf_config["gen_token_idx"] % global_inf_config["vision_adaptive_attention_update_interval"] == 0:
                global_inf_config["update_inference"] = True
        else:
            compute_part_vision_token = False


        if global_inf_config["gen_token_idx"] > 30:
            breakp = 1

        
        # ==================================================

        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # =====================================================  

        if global_inf_config["gen_token_idx"] > 0:
            position_ids = global_inf_config["decode_position_ids"].clone()
            global_inf_config["decode_position_ids"][0,0] += 1

        # 处理mask和position
        if global_inf_config["text_pruning_flag"] or global_inf_config["vision_adaptive_attention"] or global_inf_config["prefill_filter_flag"]:
            if global_inf_config["gen_token_idx"] == 0:
                input_attention_mask = causal_mask
            else:
                input_attention_mask = None
        else:
            input_attention_mask = causal_mask

        # # 处理mask和position
        # if global_inf_config["text_pruning_flag"] or global_inf_config["vision_adaptive_attention"] or global_inf_config["prefill_filter_flag"]:
        #     if global_inf_config["gen_token_idx"] == 0:
        #         input_attention_mask = causal_mask
        #     else:
        #         position_ids = global_inf_config["decode_position_ids"].clone()
        #         global_inf_config["decode_position_ids"][0,0] += 1
        #         input_attention_mask = None
        # else:
        #     input_attention_mask = causal_mask


        if global_inf_config["gen_token_idx"] == 0:
            global_inf_config["cur_text_num"] = position_ids.max().item() + 1 - global_inf_config["sys_length"] - global_inf_config["image_token_default_length"]

        else:
            global_inf_config["cur_text_num"] += 1

        actual_text_length = position_ids.max().item() + 1 - global_inf_config["sys_length"] - global_inf_config["image_token_default_length"]

        idx = -1
        # =====================================================

        

        for decoder_layer in self.layers:

            idx += 1

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:


                # ============================================
                # pre-process before decoder inference
                # ============================================

                layer_image_token_length = global_inf_config["image_token_current_length"]

                if global_inf_config["vision_adaptive_attention"] and compute_part_vision_token and global_inf_config["gen_token_idx"] > 0 and not global_inf_config["update_inference"]:
                    input_attention_mask = causal_mask.clone()
                    # input_attention_mask[:, :, :, global_inf_config["uncompute_vision_index_list"][idx]] = float('-inf')
                    input_attention_mask[:, :, :, global_inf_config["uncompute_vision_index_list"][idx]] = torch.finfo(hidden_states.dtype).min

                # ============================================




                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=input_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=True,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if True else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            
            
            # ====================================================
            # Post Process
            # ====================================================


            global_inf_config["attention_weights"][idx] = layer_outputs[1]

            if global_inf_config["all_use_h2o"]:

                if global_inf_config["gen_token_idx"] == 0:

                    seq_len_ = global_inf_config["attention_weights"][idx].shape[-1]

                    # if 'auto_half_h2o' in global_inf_config and global_inf_config['auto_half_h2o'] == True:
                    #     decoder_layer.kv_cache = H2OKVCache_LayerWise(
                    #                 hh_size=int(seq_len_* 0.4 * 0.25),
                    #                 recent_size=int(seq_len_ * 0.4 * 0.75),
                    #                 k_seq_dim=2,
                    #                 v_seq_dim=2,
                    #             )
                    # else:
                    #     decoder_layer.kv_cache = H2OKVCache_LayerWise(
                    #                 hh_size=global_inf_config["h2o_hh_size"],
                    #                 recent_size=global_inf_config["h2o_recent_size"],
                    #                 k_seq_dim=2,
                    #                 v_seq_dim=2,
                    #             )
                    
                    decoder_layer.kv_cache = H2OKVCache_LayerWise(
                                    hh_size=int(seq_len_* 0.5 * 0.25),
                                    recent_size=int(seq_len_ * 0.5 * 0.75),
                                    k_seq_dim=2,
                                    v_seq_dim=2,
                                )      

                seq_len_ = position_ids.max().item()+1

                past_key_value = (next_decoder_cache.key_cache[idx], next_decoder_cache.value_cache[idx])
                
                
                past_key_value = decoder_layer.kv_cache(past_key_value, layer_outputs[1].detach().clone(),
                                                        hh_size=int(seq_len_* 0.5 * 0.25), 
                                                        recent_size=int(seq_len_ * 0.5 * 0.75)
                                                        )
                
                # past_key_value = decoder_layer.kv_cache(past_key_value, layer_outputs[1].detach().clone())

                next_decoder_cache.key_cache[idx] = past_key_value[0]
                next_decoder_cache.value_cache[idx] = past_key_value[1]


            attention_cal_flag = False


            if global_inf_config["text_pruning_flag"]:

                if not attention_cal_flag:
                    last_layer_attention = global_inf_config["attention_weights"][idx]
                    last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                    last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                    attention_cal_flag = True

                attention_len = last_layer_attention_avg.shape[1]
                text_length = attention_len - global_inf_config["text_start_idx"]

                if global_inf_config["gen_token_idx"] == 0:
        
                    decoder_layer.text_weights = last_layer_attention.new_zeros(10240)
        
                    # past_attention = last_layer_attention_avg[:, global_inf_config["text_start_idx"]:].sum(dim=0)

                    if global_inf_config["text_use_h2o"]:
                        decoder_layer.kv_cache = H2OKVCache_LayerWise(
                                    hh_size=global_inf_config["h2o_hh_size"],
                                    recent_size=global_inf_config["h2o_recent_size"],
                                    k_seq_dim=2,
                                    v_seq_dim=2,
                                )
                    elif global_inf_config["compare_text_direct_pruning"]:
                        decoder_layer.kv_cache = TextPruneDirectKVCache(
                                    cache_size=global_inf_config["h2o_hh_size"] + global_inf_config["h2o_recent_size"],
                                    k_seq_dim=2,
                                    v_seq_dim=2,
                                )                        
                    else:        
                        # for text_idx in range(text_length-1):
                        #     decoder_layer.text_weights[text_idx:text_idx+1] = last_layer_attention_avg[global_inf_config["text_start_idx"]+text_idx, global_inf_config["text_start_idx"]+text_idx:global_inf_config["text_start_idx"]+text_idx+1]                    
                        #     cumulative_weights = last_layer_attention_avg[global_inf_config["text_start_idx"]+text_idx, global_inf_config["text_start_idx"]:global_inf_config["text_start_idx"]+text_idx+1] - last_layer_attention_avg[global_inf_config["text_start_idx"]+text_idx, global_inf_config["text_start_idx"]:global_inf_config["text_start_idx"]+text_idx+1].mean()
                        #     decoder_layer.text_weights[:text_idx] += cumulative_weights[:-1]
                        
                        decoder_layer.kv_cache = TextPruneHistoryKVCache(
                                    pruning_rate=global_inf_config['text_pruning_rate'],
                                    k_seq_dim=2,
                                    v_seq_dim=2,
                                )   
                        

                        
        
                    # weaken_tensor = torch.arange(text_length-1, -1, -1, device=past_attention.device) * global_inf_config["weaken_factor"]
                    # past_attention = past_attention - weaken_tensor
        
                    # decoder_layer.text_weights[:past_attention.shape[0]] = past_attention

                    # if global_inf_config["text_use_h2o"]:

                    #     decoder_layer.accumulated_attention_score = last_layer_attention.new_zeros(1024)
                    #     past_attention = last_layer_attention_avg[:-1, global_inf_config["text_start_idx"]:].sum(dim=0)
                    #     decoder_layer.accumulated_attention_score[:len(past_attention)] = past_attention


                # if global_inf_config["text_use_h2o"]:
                #     past_attention = last_layer_attention_avg_last_tok[global_inf_config["text_start_idx"]:]
                #     decoder_layer.accumulated_attention_score[:len(past_attention)] += past_attention

                # if not global_inf_config["text_use_h2o"] and not global_inf_config["compare_text_direct_pruning"]:
                #     decoder_layer.text_weights[text_length-1:text_length] = last_layer_attention_avg_last_tok[-1]
                #     cumulative_weights = last_layer_attention_avg_last_tok[global_inf_config["text_start_idx"]:] - last_layer_attention_avg_last_tok[global_inf_config["text_start_idx"]:].mean()
                #     decoder_layer.text_weights[:text_length-1] += cumulative_weights[:-1]


                # if global_inf_config["cur_text_num"] >= global_inf_config["text_pruning_interval"]:
                if True:
        
                    if global_inf_config["text_use_h2o"]:

                        # cur_text_weights = decoder_layer.accumulated_attention_score[:text_length]

                        # text_remain_indices = cur_text_weights.topk(ATTENTION_RANK).indices
                        # text_remain_indices = text_remain_indices.sort().values
                        # text_remain_indices_global = text_remain_indices + global_inf_config["text_start_idx"]
                
                        # keep_indexs = torch.cat( (torch.arange(global_inf_config["text_start_idx"], device=text_remain_indices_global.device), 
                        #                           text_remain_indices_global) 
                        #                           )
                        
                        # next_decoder_cache.key_cache[idx] = next_decoder_cache.key_cache[idx][:, :, keep_indexs, :]
                        # next_decoder_cache.value_cache[idx] = next_decoder_cache.value_cache[idx][:, :, keep_indexs, :]

                        # weights_mask = torch.zeros_like(decoder_layer.accumulated_attention_score, dtype=torch.bool, device=decoder_layer.accumulated_attention_score.device)
                        # weights_mask[text_remain_indices] = True
                        # weights_mask[text_length:] = True
    
                        # decoder_layer.accumulated_attention_score = decoder_layer.accumulated_attention_score[weights_mask]


                        ATTENTION_RANK = round((1 - global_inf_config["text_pruning_rate"]) * actual_text_length)

                        ATTENTION_RANK = max(ATTENTION_RANK, global_inf_config["text_init_cache"])


                        text_start_idx = global_inf_config["text_start_idx"]
    
                        past_key_value = (next_decoder_cache.key_cache[idx][:, :, text_start_idx:, :], next_decoder_cache.value_cache[idx][:, :, text_start_idx:, :])
    
                        if global_inf_config['gen_token_idx'] == 0:
                            text_raw_attention = last_layer_attention[:, :, text_start_idx:, text_start_idx:]
                        else:
                            text_raw_attention = last_layer_attention[:, :, :, text_start_idx:]

                        hh_size = int(ATTENTION_RANK * global_inf_config["text_h20_hh_rate_in_cache"])
                        recent_size = ATTENTION_RANK - hh_size
    
                        past_key_value = decoder_layer.kv_cache(past_key_value, text_raw_attention.detach().clone(),
                                                                hh_size=hh_size,
                                                                recent_size=recent_size)
                
                        next_decoder_cache.key_cache[idx] = torch.cat( (
                            next_decoder_cache.key_cache[idx][:, :, :text_start_idx, :],
                            past_key_value[0]
                        ), dim=-2 )
    
                        next_decoder_cache.value_cache[idx] = torch.cat( (
                            next_decoder_cache.value_cache[idx][:, :, :text_start_idx, :],
                            past_key_value[1]
                        ), dim=-2 )
                        

                    elif global_inf_config["compare_text_direct_pruning"]:

                        # 直接对比

                        # text_attention = last_layer_attention_avg_last_tok[global_inf_config["text_start_idx"]:]
                        # cur_text_weights = text_attention

                        # text_remain_indices = cur_text_weights.topk(ATTENTION_RANK).indices
                        # text_remain_indices = text_remain_indices.sort().values
                        # text_remain_indices_global = text_remain_indices + global_inf_config["text_start_idx"]
                
                        # keep_indexs = torch.cat( (torch.arange(global_inf_config["text_start_idx"], device=text_remain_indices_global.device), 
                        #                           text_remain_indices_global) 
                        #                           )
                        
                        # next_decoder_cache.key_cache[idx] = next_decoder_cache.key_cache[idx][:, :, keep_indexs, :]
                        # next_decoder_cache.value_cache[idx] = next_decoder_cache.value_cache[idx][:, :, keep_indexs, :]

                        ATTENTION_RANK = round((1 - global_inf_config["text_pruning_rate"]) * actual_text_length)

                        ATTENTION_RANK = max(ATTENTION_RANK, global_inf_config["text_init_cache"])

                        if ATTENTION_RANK <= text_length:

                            text_start_idx = global_inf_config["text_start_idx"]
    
                            past_key_value = (next_decoder_cache.key_cache[idx][:, :, text_start_idx:, :], next_decoder_cache.value_cache[idx][:, :, text_start_idx:, :])
    
                            if global_inf_config['gen_token_idx'] == 0:
                                text_raw_attention = last_layer_attention[:, :, text_start_idx:, text_start_idx:]
                            else:
                                text_raw_attention = last_layer_attention[:, :, :, text_start_idx:]
    
                            past_key_value = decoder_layer.kv_cache(past_key_value, text_raw_attention.detach().clone(),
                                                                    cache_size=ATTENTION_RANK)
                
                            next_decoder_cache.key_cache[idx] = torch.cat( (
                                next_decoder_cache.key_cache[idx][:, :, :text_start_idx, :],
                                past_key_value[0]
                            ), dim=-2 )
    
                            next_decoder_cache.value_cache[idx] = torch.cat( (
                                next_decoder_cache.value_cache[idx][:, :, :text_start_idx, :],
                                past_key_value[1]
                            ), dim=-2 )

                    else:

                        # 自己的方法

                        ATTENTION_RANK = round((1 - global_inf_config["text_pruning_rate"]) * actual_text_length)

                        ATTENTION_RANK = max(ATTENTION_RANK, global_inf_config["text_init_cache"])

                        text_start_idx = global_inf_config["text_start_idx"]
    
                        past_key_value = (next_decoder_cache.key_cache[idx][:, :, text_start_idx:, :], next_decoder_cache.value_cache[idx][:, :, text_start_idx:, :])
    
                        if global_inf_config['gen_token_idx'] == 0:
                            text_raw_attention = last_layer_attention[:, :, text_start_idx:, text_start_idx:]
                        else:
                            text_raw_attention = last_layer_attention[:, :, :, text_start_idx:]

    
                        past_key_value = decoder_layer.kv_cache(past_key_value, text_raw_attention.detach().clone(),
                                                                cache_size=ATTENTION_RANK)
                
                        next_decoder_cache.key_cache[idx] = torch.cat( (
                            next_decoder_cache.key_cache[idx][:, :, :text_start_idx, :],
                            past_key_value[0]
                        ), dim=-2 )
    
                        next_decoder_cache.value_cache[idx] = torch.cat( (
                            next_decoder_cache.value_cache[idx][:, :, :text_start_idx, :],
                            past_key_value[1]
                        ), dim=-2 )
    
                        # if ATTENTION_RANK <= text_length:
 
                        #     cur_text_weights = 0.2 * decoder_layer.text_weights[:text_length] + last_layer_attention_avg_last_tok[global_inf_config["text_start_idx"]:]
                            
                        #     text_remain_indices = cur_text_weights.topk(ATTENTION_RANK).indices
                        #     text_remain_indices = text_remain_indices.sort().values
                        #     text_remain_indices_global = text_remain_indices + global_inf_config["text_start_idx"]
    
                        #     keep_indexs = torch.cat( (torch.arange(global_inf_config["text_start_idx"], device=text_remain_indices_global.device), 
                        #                               text_remain_indices_global) 
                        #                               )
                            
                        #     next_decoder_cache.key_cache[idx] = next_decoder_cache.key_cache[idx][:, :, keep_indexs, :]
                        #     next_decoder_cache.value_cache[idx] = next_decoder_cache.value_cache[idx][:, :, keep_indexs, :]
        
                        #     weights_mask = torch.zeros_like(decoder_layer.text_weights, dtype=torch.bool, device=decoder_layer.text_weights.device)
                        #     weights_mask[text_remain_indices] = True
                        #     weights_mask[text_length:] = True
        
                        #     decoder_layer.text_weights = decoder_layer.text_weights[weights_mask]


            if global_inf_config["vision_adaptive_attention"]:

                ATTENTION_RANK = int((1 - global_inf_config["vision_adaptive_attention_rate"]) * global_inf_config["image_token_default_length"])
                COMPUTE_ATTENTION_RANK = int((1 - global_inf_config["vision_adaptive_attention_compute_rate"]) * global_inf_config["image_token_default_length"])

                if global_inf_config["gen_token_idx"] == 0:

                    if not attention_cal_flag:
                        last_layer_attention = global_inf_config["attention_weights"][idx]
                        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                        attention_cal_flag = True

                    last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+layer_image_token_length]

                    top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH

                    top_attention_rank_index_sort = top_attention_rank_index.sort()

                    top_attention_rank_index = top_attention_rank_index_sort.values

                    process_img_token_idxs = torch.argsort(top_attention_rank_index_sort.indices)
                    
                    global_inf_config["compute_vision_index_list"][idx] = process_img_token_idxs[:COMPUTE_ATTENTION_RANK] + SYS_LENGTH
                    global_inf_config["uncompute_vision_index_list"][idx] = process_img_token_idxs[COMPUTE_ATTENTION_RANK:] + SYS_LENGTH

                    keep_indexs = torch.cat( (torch.arange(SYS_LENGTH,device=top_attention_rank_index.device), 
                                          top_attention_rank_index, 
                                          torch.arange(SYS_LENGTH+layer_image_token_length, next_decoder_cache.key_cache[idx].size(2), device=top_attention_rank_index.device)
                                          ) )
                    
                    next_decoder_cache.key_cache[idx] = next_decoder_cache.key_cache[idx][:, :, keep_indexs, :]
                    next_decoder_cache.value_cache[idx] = next_decoder_cache.value_cache[idx][:, :, keep_indexs, :]

                    layer_image_token_length = ATTENTION_RANK

                elif global_inf_config["update_inference"]:


                    if not attention_cal_flag:
                        last_layer_attention = global_inf_config["attention_weights"][idx]
                        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                        attention_cal_flag = True

                    last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+layer_image_token_length]

                    top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH

                    top_attention_rank_index_sort = top_attention_rank_index.sort()

                    top_attention_rank_index = top_attention_rank_index_sort.values

                    process_img_token_idxs = torch.argsort(top_attention_rank_index_sort.indices)
                    
                    global_inf_config["compute_vision_index_list"][idx] = process_img_token_idxs[:COMPUTE_ATTENTION_RANK] + SYS_LENGTH
                    global_inf_config["uncompute_vision_index_list"][idx] = process_img_token_idxs[COMPUTE_ATTENTION_RANK:] + SYS_LENGTH


            if global_inf_config["prefill_filter_flag"] and global_inf_config["gen_token_idx"] == 0:
                
                if idx == global_inf_config["prefill_filter_layer"] - 1:

                    if not attention_cal_flag:
                        last_layer_attention = global_inf_config["attention_weights"][idx]
                        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                        attention_cal_flag = True
                    
                    IMAGE_TOKEN_LENGTH = global_inf_config["image_token_default_length"]
                    ATTENTION_RANK = int((1 - global_inf_config["prefill_filter_rate"]) * IMAGE_TOKEN_LENGTH)
                    last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]

                    top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH

                    keep_indexs = torch.cat( (torch.arange(SYS_LENGTH, device=top_attention_rank_index.device), 
                                                           top_attention_rank_index, 
                                                           torch.arange(SYS_LENGTH+IMAGE_TOKEN_LENGTH, hidden_states.size(1), device=top_attention_rank_index.device)
                                                           ))

                    keep_indexs = keep_indexs.sort().values
                    new_seq_length = keep_indexs.shape[0]
                    hidden_states = hidden_states[:,keep_indexs,:]
                    # cache_position = cache_position[keep_indexs]
                    
                    position_ids = keep_indexs.unsqueeze(0)

                    attention_mask = attention_mask.new_ones(hidden_states.size(0), hidden_states.size(1))

                    temp_cache_position = torch.arange(new_seq_length, device=cache_position.device)

                    causal_mask = self._update_causal_mask(attention_mask, hidden_states, temp_cache_position)
                    input_attention_mask = causal_mask

                    global_inf_config["image_token_current_length"] = ATTENTION_RANK
                    global_inf_config["text_start_idx"] = global_inf_config["sys_length"] + global_inf_config["image_token_current_length"]

            global_inf_config["attention_weights"][idx] = None

            # ====================================================











        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        






        # ================================================
        # Inference Step Process
        # ================================================
        if global_inf_config["gen_token_idx"] == 0:
            global_inf_config["decode_position_ids"] = torch.tensor([position_ids.max().item() + 1], dtype=torch.long, device=position_ids.device).unsqueeze(0)

        if global_inf_config["cur_text_num"] >= global_inf_config["text_pruning_interval"]:
            global_inf_config["cur_text_num"] = 0

        global_inf_config["update_inference"] = False
        global_inf_config["image_token_current_length"] = layer_image_token_length
        global_inf_config["text_start_idx"] = global_inf_config["sys_length"] + global_inf_config["image_token_current_length"]

        # ================================================
        




        if global_inf_config["gen_token_idx"] >= 276:
            breakp = 2





        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(self.layers[0].self_attn, "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else cache_position[-1] + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
