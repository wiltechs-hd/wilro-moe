"""
ExpertLayer — trainable Llama-style decoder layer parallel to a frozen VLM layer.

Structure mirrors SmolLM2's LlamaDecoderLayer (RMSNorm + GQA attention + SwiGLU
FFN) so that, at each VLM depth, expert tokens and VLM tokens can be processed
with structurally identical math but separate (trainable) weights.

Only the per-side projections (q/k/v/o for attention, gate/up/down for the
SwiGLU FFN) and the RMSNorm scales live here. The actual joint-attention math
(concatenating Q/K/V across the two sides, RoPE, scaled dot product) lives in
the main model file because it needs both VLM and expert tensors at once.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMS normalisation matching Llama's implementation (no centering)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability, cast back at the end.
        orig_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(orig_dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN: (x · gate_proj) ⊙ SiLU(x · up_proj) → down_proj."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Llama convention: down(silu(gate(x)) * up(x))
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertProjections(nn.Module):
    """
    Holds the trainable QKV/O projections and FFN for the expert half of a
    joint-attention layer. Layout matches a Llama decoder layer's `self_attn`
    and `mlp` submodules so caller code can mirror VLM layer access patterns.

    Args:
        hidden_size: must match the VLM hidden dim (joint attention concats
            K/V across modalities, so dims must agree).
        num_heads:   query heads for the expert.
        num_kv_heads: key/value heads (≤ num_heads for grouped-query attention).
        head_dim:    per-head dimension (typically hidden_size // num_heads).
        intermediate_size: SwiGLU expansion width.
        rms_norm_eps: epsilon for both RMSNorms.
        dropout:     applied to attention output and FFN output (Llama itself
                     uses 0 here; default 0.1 to combat the same overfit we
                     hit with the encoder-decoder model).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size

        # ---- Self-attention projections (trainable, separate from VLM) ----
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # ---- Norms (pre-norm, Llama style) ----
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # ---- FFN (SwiGLU) ----
        self.mlp = SwiGLU(hidden_size, intermediate_size)

        # ---- Dropout ----
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)

        # ---- Init: small std so expert starts as ~identity in the residual ----
        # The o_proj zero-init is a "safe start" — initial expert contribution
        # to the residual stream is zero, so the model behaves like the frozen
        # VLM at step 0 and learns to *add* useful expert signal gradually.
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.mlp.down_proj.weight)
