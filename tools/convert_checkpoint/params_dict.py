# The simple map of names for "automated" rules.
megatron_to_llama = {
    "self_attention.dense": ".attention.wo.",
    "mlp.dense_h_to_4h": ".feed_forward.w1.",
    "mlp.dense_4h_to_h": ".feed_forward.w2.",
    "mlp.dense_proj": ".feed_forward.w3."
}

megatron_to_llama2 = {
    "self_attention.dense": ".attention.wo.",
    "mlp.dense_h_to_4h": ".feed_forward.w1.",
    "mlp.dense_4h_to_h": ".feed_forward.w2.",
    "mlp.dense_proj": ".feed_forward.w3."
}

megatron_to_transformers = {
    "attention.dense": ".attn.c_proj.",
    "self_attention.dense": ".attn.c_proj.",
    "mlp.dense_h_to_4h": ".mlp.c_fc.",
    "mlp.dense_4h_to_h": ".mlp.c_proj.",
}

# Layers need to split aross tensor parallel
tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_proj.weight",
    # "mlp.dense_4h_to_h.bias",
    # deprecated
    "attention.query_key_value.weight",
    "attention.query_key_value.bias",
    "attention.dense.weight",
    # transformers layers to split across tp ranks
    # "attn.c_attn.weight",
    # "attn.c_attn.bias",
    "self_attn.out_proj.weight",
    "fc1.weight",
    "fc2.weight",
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    # LLaMA layers to split across tp ranks
    "attention.wo.weight",
    "feed_forward.w1.weight",
    "feed_forward.w2.weight",
    "feed_forward.w3.weight",
]