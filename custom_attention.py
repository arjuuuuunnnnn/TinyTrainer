from transformers.integrations.sdpa_attention import repeat_kv as original_repeat_kv

def custom_repeat_kv(hidden_states, num_key_value_groups):
    if hidden_states.dim() != 4:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states.view(batch, num_key_value_heads, slen, head_dim)
    return original_repeat_kv(hidden_states, num_key_value_groups)

# Override
import transformers.integrations.sdpa_attention
transformers.integrations.sdpa_attention.repeat_kv = custom_repeat_kv
