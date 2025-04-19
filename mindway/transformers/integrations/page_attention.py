from typing import Optional

import math
import mindspore as ms
from mindspore import nn
from transformers import PretrainedConfig

from mindway.transformers.mindspore_adapter import str_to_dtype
from mindway.transformers.mindspore_adapter.infer_attention import InferAttention


class PageAttention(nn.Cell):
    """
    Qwen3 page attention module
    """

    def __init__(self, config: PretrainedConfig, layer_idx: Optional[int] = None):
        super().__init__()
        compute_dtype = str_to_dtype(config.mindspore_dtype)

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.q_proj = nn.Dense(
            config.hidden_size, config.num_attention_heads * self.head_dim, has_bias=config.attention_bias
        )
        self.k_proj = nn.Dense(
            config.hidden_size, config.num_key_value_heads * self.head_dim, has_bias=config.attention_bias
        )
        self.v_proj = nn.Dense(
            config.hidden_size, config.num_key_value_heads * self.head_dim, has_bias=config.attention_bias
        )
        self.o_proj = nn.Dense(
            config.num_attention_heads * self.head_dim, config.hidden_size, has_bias=config.attention_bias
        )

        self.infer_attention = InferAttention(
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
            config.num_key_value_heads,
            seq_length=config.max_position_embeddings,
            pa_n_head_split=config.num_attention_heads,
            pa_n_kv_head_split=config.hidden_size // config.num_attention_heads,
            scale_value=1.0 / (math.sqrt(config.hidden_size // config.num_attention_heads)),
            pre_tokens=2147483647,
            next_tokens=0,
            block_size=32,
            num_blocks=1024,
            is_dynamic=True,
            use_flash_attention=True,
            rotary_cos_format=2,
            compute_dtype=compute_dtype,
        )

        self.is_first_iteration = True

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[tuple[ms.Tensor, ms.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[ms.Tensor] = None,
        block_tables: Optional[ms.Tensor] = None,
        slot_mapping: Optional[ms.Tensor] = None,
        freqs_cis: Optional[ms.Tensor] = None,
        mask: Optional[ms.Tensor] = None,
        batch_valid_length: Optional[ms.Tensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        attn_output = self.infer_attention(
            query_states,
            key_states,
            value_states,
            batch_valid_length,
            block_tables,
            slot_mapping,
            freqs_cis,
            mask,
            q_seq_lens=None,
        )

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value