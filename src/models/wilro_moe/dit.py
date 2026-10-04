"""The DiT block every expert in WILRO-MoE stacks.

Self-attention over the action tokens, cross-attention into one band of the
frozen VLM's KV cache, an optional cross-attention into robot-camera tokens,
and an adaLN-Zero-modulated SwiGLU FFN.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLU
from .smolvlm_encoder import _modulate


class DiTLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.1,
        use_vision_ca: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_vision_ca = use_vision_ca

        # ── Self-attention (over DiT sequence) ──────────────────────────
        self.sa_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.sa_q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.sa_k = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.sa_v = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.sa_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.sa_drop = nn.Dropout(dropout)

        # ── Cross-attention (Q from DiT, K/V from VLM KV cache) ─────────
        self.ca_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ca_q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.ca_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.ca_drop = nn.Dropout(dropout)

        # ── Robot CNN cross-attention (Q from DiT, K/V from Robot CNN) ──
        # This enables direct high-resolution spatial grounding: action
        # queries can attend to Robot CNN's fine-grained feature map
        # (14x14 @ 224x224) instead of only VLM's coarse SigLIP patches
        # (~729 patches @ 384x384). Critical for precise object localization
        # in spatial reasoning tasks (e.g., "bowl closer to plate").
        if use_vision_ca:
            self.robot_ca_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.robot_ca_q = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.robot_ca_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
            self.robot_ca_drop = nn.Dropout(dropout)

        # ── FFN ─────────────────────────────────────────────────────────
        self.ffn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.ffn = SwiGLU(hidden_size, intermediate_size)
        self.ffn_drop = nn.Dropout(dropout)

        # ── adaLN-Zero: 12 modulation vectors (shift/scale/gate × 4) ────
        # With robot_ca: 4 sublayers (sa, ca, robot_ca, ffn) × 3 = 12
        # Without robot_ca: 3 sublayers (sa, ca, ffn) × 3 = 9
        adaLN_dim = 12 * hidden_size if use_vision_ca else 9 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, adaLN_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        vlm_k: torch.Tensor,
        vlm_v: torch.Tensor,
        vlm_kv_pad_mask: Optional[torch.Tensor],
        self_attn_mask: torch.Tensor,
        vision_k: Optional[torch.Tensor] = None,
        vision_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:               (B, L_dit, H)
        t_emb:           (B, H) — per-batch time conditioning
        vlm_k, vlm_v:    (B, num_kv_heads, L_vlm, head_dim) — frozen VLM cache
        vlm_kv_pad_mask: (B, L_vlm) bool, True at valid VLM positions
        self_attn_mask:  (1, 1, L_dit, L_dit) additive mask
        vision_k, vision_v: (B, num_kv_heads, R, head_dim) — Robot CNN K/V for
                         high-resolution spatial cross-attention (optional)
        """
        B, L_dit, _ = x.shape

        mod = self.adaLN_modulation(t_emb)
        if self.use_vision_ca and vision_k is not None:
            # 12 chunks: sa(3), ca(3), robot_ca(3), ffn(3)
            (
                s_sa, sc_sa, g_sa,
                s_ca, sc_ca, g_ca,
                s_rca, sc_rca, g_rca,
                s_ff, sc_ff, g_ff,
            ) = mod.chunk(12, dim=-1)
        else:
            # 9 chunks: sa(3), ca(3), ffn(3)
            (
                s_sa, sc_sa, g_sa,
                s_ca, sc_ca, g_ca,
                s_ff, sc_ff, g_ff,
            ) = mod.chunk(9, dim=-1)

        # ── Self-attention ───────────────────────────────────────────
        h = _modulate(self.sa_norm(x), s_sa, sc_sa)
        Q = self.sa_q(h).view(B, L_dit, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.sa_k(h).view(B, L_dit, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.sa_v(h).view(B, L_dit, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(r, dim=1)
            V = V.repeat_interleave(r, dim=1)
        sa = F.scaled_dot_product_attention(Q, K, V, attn_mask=self_attn_mask, is_causal=False)
        sa = sa.transpose(1, 2).contiguous().view(B, L_dit, self.num_heads * self.head_dim)
        sa = self.sa_drop(self.sa_o(sa))
        x = x + g_sa.unsqueeze(1) * sa

        # ── Cross-attention to frozen VLM cache ──────────────────────
        h = _modulate(self.ca_norm(x), s_ca, sc_ca)
        Q = self.ca_q(h).view(B, L_dit, self.num_heads, self.head_dim).transpose(1, 2)
        Kv, Vv = vlm_k, vlm_v
        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            Kv = Kv.repeat_interleave(r, dim=1)
            Vv = Vv.repeat_interleave(r, dim=1)
        if vlm_kv_pad_mask is not None:
            kpad = ~vlm_kv_pad_mask                                 # True = pad
            ca_mask = torch.zeros(B, 1, 1, vlm_kv_pad_mask.shape[-1],
                                  device=x.device, dtype=Q.dtype)
            ca_mask.masked_fill_(kpad.unsqueeze(1).unsqueeze(1), float("-inf"))
        else:
            ca_mask = None
        ca = F.scaled_dot_product_attention(Q, Kv, Vv, attn_mask=ca_mask, is_causal=False)
        ca = ca.transpose(1, 2).contiguous().view(B, L_dit, self.num_heads * self.head_dim)
        ca = self.ca_drop(self.ca_o(ca))
        x = x + g_ca.unsqueeze(1) * ca

        # ── Robot CNN cross-attention (high-res spatial grounding) ───
        if self.use_vision_ca and vision_k is not None:
            h = _modulate(self.robot_ca_norm(x), s_rca, sc_rca)
            Q = self.robot_ca_q(h).view(B, L_dit, self.num_heads, self.head_dim).transpose(1, 2)
            Kr, Vr = vision_k, vision_v
            if self.num_kv_heads != self.num_heads:
                r = self.num_heads // self.num_kv_heads
                Kr = Kr.repeat_interleave(r, dim=1)
                Vr = Vr.repeat_interleave(r, dim=1)
            robot_ca = F.scaled_dot_product_attention(Q, Kr, Vr, is_causal=False)
            robot_ca = robot_ca.transpose(1, 2).contiguous().view(B, L_dit, self.num_heads * self.head_dim)
            robot_ca = self.robot_ca_drop(self.robot_ca_o(robot_ca))
            x = x + g_rca.unsqueeze(1) * robot_ca

        # ── FFN ──────────────────────────────────────────────────────
        h = _modulate(self.ffn_norm(x), s_ff, sc_ff)
        ff = self.ffn_drop(self.ffn(h))
        x = x + g_ff.unsqueeze(1) * ff

        return x


