"""Model shape constants mirrored from ds4.c (shape tables around ds4.c:700-916).

Kernels are generated for the concrete sizes listed here; ds4 dispatches to
the specialised variant and falls back to its hand-written kernel otherwise.
"""

# DeepSeek V4 Flash (g_ds4_shape, ds4.c:880-916)
DSV4_FLASH = dict(
    n_embd=4096,
    n_head=64,
    n_head_kv=1,
    head_dim=512,
    n_rot=64,
    n_expert=256,
    n_expert_used=6,
    n_shared_expert=1,
    ff_exp=2048,
    swa=128,
    indexer_heads=64,
    indexer_dim=128,
    top_k=512,
    n_hc=4,
    sinkhorn_iters=20,
)

# Row widths that RMSNorm is applied to across the supported models.
RMS_NORM_WIDTHS = (
    4096,   # n_embd (DSV4, GLM)
    16384,  # n_embd * n_hc: the plain norm of the flattened hyper-connection state
    512,    # head_dim / kv_lora
    2048,   # ff_exp mid
    128,    # indexer_dim
    1024,   # q_lora rank (q_a norm) and the DSpark drafter's hidden size
)

# Input widths of the dense Q8_0 projections hit by single-token decode.
Q8_0_GEMV_IN_DIMS = (
    4096,  # n_embd -> q_a / kv / router inputs
    1024,  # attn_output_a rank -> attn_output_b
    2048,  # shared expert down (ff_exp -> n_embd)
    8192,  # attn_output_b as a plain GEMV (1-token batch paths, e.g. DSpark's verifier)
    12288, # DSpark support model main_proj (-> 4096)
    256,   # DSpark Markov heads (-> vocab)
)

# Shared expert gate/up input width (n_embd).
Q8_0_SHARED_IN_DIMS = (
    4096,
)

# Input width of attn_output_b (n_groups * rank), fused with the HC expand.
Q8_0_HC_EXPAND_IN_DIMS = (
    8192,
)

# Per-group input width of attn_output_a (head groups x (group_dim -> rank)).
Q8_0_GROUPED_IN_DIMS = (
    4096,
)

# Routed expert input width (n_embd) for the MoE decode kernels.
MOE_IN_DIMS = (
    4096,
)

# Routed expert hidden width (ff_exp) for the down projection.
MOE_MID_DIMS = (
    2048,
)

# Input widths of the dense Q8_0 prefill GEMMs (attn_q_a, attn_q_b via rank,
# shared gate/up, shared down, attn_output_b).
Q8_0_GEMM_IN_DIMS = (
    4096,
    1024,
    2048,
    8192,
    12288,  # DSpark support model main_proj (batched drafter passes)
    256,    # DSpark Markov heads
)

# Input widths of the f16 prefill GEMMs (compressor projections, indexer
# proj and q_b, router gate_inp). Must be a subset of Q8_0_GEMM_IN_DIMS
# (the f16 activation conversion kernels are shared).
F16_GEMM_IN_DIMS = (
    4096,
    1024,
)
