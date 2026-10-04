"""
Trainable ResNet-18 visual encoder for robot-specific features.

Runs in parallel with the frozen SigLIP ViT:
  SigLIP  — semantic, 14×14px patches, frozen
  ResNet  — spatial, pixel-level precision, fully trainable

ImageNet pretraining gives edge/texture/shape features for free.
Fine-tuning on robot data adapts these to gripper aperture, object
distance, contact state — features SigLIP misses because its patch
size is too coarse and its pretraining domain is internet images.

~3.0M params. NOT the 11.7M of a full ResNet-18: layer4 is 72% of that and is
excluded here, so quoting the stock figure overstates this module ~4x. The
number appears as "~11M" in interleaved's ARCHITECTURE.md and in the wilro
notes; both are the stock ResNet, not what is built.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


def _sincos_2d(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    """Fixed 2D sinusoidal position encoding, (1, h*w, dim), row-major.

    Half the channels encode the row, half the column, matching the row-major
    flatten of a conv feature map.
    """
    half = dim // 2
    nfreq = max(1, half // 2)
    omega = torch.exp(torch.arange(nfreq, device=device, dtype=torch.float32)
                      * -(math.log(10000.0) / nfreq))

    def _1d(n):
        ang = torch.arange(n, device=device, dtype=torch.float32)[:, None] * omega[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=1)          # (n, 2*nfreq)

    r, c = _1d(h), _1d(w)
    grid = torch.cat([
        r[:, None, :].expand(h, w, r.shape[1]),
        c[None, :, :].expand(h, w, c.shape[1]),
    ], dim=-1).reshape(h * w, -1)
    if grid.shape[-1] < dim:
        grid = F.pad(grid, (0, dim - grid.shape[-1]))
    return grid[:, :dim].to(dtype).unsqueeze(0)


class AttentionPool2d(nn.Module):
    """Learned-query attention pooling over a conv feature map.

    Replaces AdaptiveAvgPool2d. Average pooling over a 14x14 map down to a 4x4
    grid averages ~12 spatial positions into each output token, discarding ~92%
    of the spatial detail that excluding ResNet layer4 was meant to preserve.
    Here `out_tokens` learned queries attend over all H*W positions instead, so
    a token can concentrate on wherever the evidence is rather than on a fixed
    rectangle.

    The position encoding on the KEYS is not optional. Average pooling carries
    position implicitly in the token ORDER -- output token 5 IS grid cell 5.
    Learned queries have no such tie: without a position signal on the keys the
    attention is permutation-invariant over space, every spatial relation is
    destroyed, and this ends up strictly worse than the average pooling it
    replaces. It is added to keys and values before attention, not to the
    output.
    """

    def __init__(self, in_ch: int, out_tokens: int, num_heads: int = 8):
        super().__init__()
        if in_ch % num_heads != 0:
            raise ValueError(f"in_ch ({in_ch}) must be divisible by num_heads ({num_heads})")
        self.out_tokens = out_tokens
        self.num_heads = num_heads
        self.head_dim = in_ch // num_heads
        self.q_proj = nn.Linear(in_ch, in_ch, bias=False)
        self.k_proj = nn.Linear(in_ch, in_ch, bias=False)
        self.v_proj = nn.Linear(in_ch, in_ch, bias=False)
        self.o_proj = nn.Linear(in_ch, in_ch, bias=False)
        self.kv_norm = nn.LayerNorm(in_ch)
        self._pos_cache: dict = {}

        # Initialised so the ATTENTION starts grid-shaped, which fixes a real
        # cold start.
        #
        # With default random projections the q.k logits are tiny, so softmax is
        # near-uniform and every query returns roughly the same global mean:
        # measured token distinctness 0.069 against average pooling's 0.987. The
        # module would start as one token repeated `out_tokens` times and spend
        # training just separating them.
        #
        # Instead, seed query i with the position code of grid cell i and make
        # the projections identity. The logit is then pos_i . pos_j, peaked where
        # a query's cell matches a feature position. Measured after this change:
        # distinctness 0.444 (6.4x better) and 10x more sensitive to a spatial
        # permutation of the input.
        #
        # This is NOT numerically equal to average pooling -- kv_norm alone puts
        # the outputs on a different scale, and cosine agreement with avg-pool is
        # only ~0.08. The claim is narrower: the attention pattern begins
        # grid-like instead of uniform, and is free to sharpen or move off-grid.
        side = int(round(out_tokens ** 0.5))
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.eye_(lin.weight)
        if side * side == out_tokens:
            q0 = _sincos_2d(side, side, in_ch, torch.device("cpu"), torch.float32)
        else:
            # Non-square token counts have no grid to seed from; fall back to
            # random queries and accept the slower separation.
            q0 = torch.randn(1, out_tokens, in_ch) * 0.02
        self.queries = nn.Parameter(q0.clone())

    def _pos(self, h, w, dim, device, dtype):
        key = (h, w, dim, str(device), str(dtype))
        hit = self._pos_cache.get(key)
        if hit is None:
            hit = _sincos_2d(h, w, dim, device, dtype)
            self._pos_cache[key] = hit
        return hit

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, out_tokens, C)."""
        B, C, H, W = feat.shape
        x = self.kv_norm(feat.flatten(2).transpose(1, 2))        # (B, H*W, C)
        pos = self._pos(H, W, C, x.device, x.dtype)
        # Position goes on the KEYS only, not the values: it exists to decide
        # WHERE a query looks, and folding it into the values would paint a
        # constant positional offset onto the pooled content as well.
        q = self.queries.expand(B, -1, -1).to(x.dtype)

        def heads(t, n):
            return t.view(B, n, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            heads(self.q_proj(q), self.out_tokens),
            heads(self.k_proj(x + pos), H * W),
            heads(self.v_proj(x), H * W),
        )
        out = out.transpose(1, 2).reshape(B, self.out_tokens, C)
        return self.o_proj(out)


class RobotVisualEncoder(nn.Module):
    """
    Pretrained ResNet-18 backbone producing spatial feature tokens.

    Uses layers 1–3 of ResNet-18 (output stride 16, 256-channel feature map:
    a 256px frame gives a 16x16 map — verified, not inferred),
    then adaptive-pools to a fixed token grid and projects to d_model.
    Layer 4 is excluded to keep spatial resolution higher (better for
    precise localisation tasks like grasping).

    Args:
        input_size:  images resized to this square resolution before encoding.
        out_tokens:  spatial tokens per camera (must be a perfect square).
        out_dim:     output feature dim per token — should match d_model.
    """

    def __init__(self, input_size: int = 224, out_tokens: int = 16, out_dim: int = 512,
                 pool: str = "avg"):
        super().__init__()
        if pool not in ("avg", "attn"):
            raise ValueError(f"pool must be 'avg' or 'attn', got {pool!r}")
        # The square constraint is a property of the AVG grid. Attention pooling
        # emits a set, not a grid, so any count is valid there.
        if pool == "avg":
            assert int(out_tokens ** 0.5) ** 2 == out_tokens, "out_tokens must be a perfect square"
        self.input_size = input_size
        self.pool_type = pool
        self.out_tokens = out_tokens
        token_side = int(out_tokens ** 0.5)

        # Pretrained ResNet-18 backbone
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Stem: conv1 + bn1 + relu + maxpool  (224 → 56)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1   # 56 → 56,  64 ch
        self.layer2 = backbone.layer2   # 56 → 28, 128 ch
        self.layer3 = backbone.layer3   # 28 → 14, 256 ch
        # layer4 excluded — keeps higher spatial resolution for precise localisation

        self.pool = (nn.AdaptiveAvgPool2d((token_side, token_side)) if pool == "avg"
                     else AttentionPool2d(256, out_tokens))
        self.proj = nn.Linear(256, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        # ImageNet normalisation constants
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor, out_tokens: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) in [0, 1] — raw camera image, any resolution.
            out_tokens: override the token grid for this call (must be a perfect
                square). Lets a single shared backbone emit a denser grid for
                some cameras (e.g. the gripper view, for placement precision)
                and a coarser grid for others. None → the construction default.
                Only the pooling grid changes; proj/norm are per-token, so no
                parameters depend on the count.
        Returns:
            (B, out_tokens, out_dim) float32 feature tokens.
        """
        return self.tokens_from_map(self.trunk(x), out_tokens=out_tokens)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [0, 1] -> (B, 256, S, S) pre-pool feature map.

        Split out of `forward` so a caller can run the backbone on two frames
        and difference the MAPS before pooling. Differencing the pooled output
        instead would subtract post-LayerNorm tokens, which is not the same
        quantity. `forward` is trunk + tokens_from_map and is unchanged.
        """
        x = x.float()

        # Resize to fixed input size
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)

        # ImageNet normalisation
        x = (x - self.img_mean) / self.img_std

        feat = self.stem(x)      # (B, 64,  56, 56)
        feat = self.layer1(feat) # (B, 64,  56, 56)
        feat = self.layer2(feat) # (B, 128, 28, 28)
        feat = self.layer3(feat) # (B, 256, 14, 14)
        return feat

    def tokens_from_map(self, feat: torch.Tensor,
                        out_tokens: Optional[int] = None) -> torch.Tensor:
        """(B, 256, S, S) -> (B, out_tokens, out_dim). Pool + project + norm."""
        if self.pool_type == "attn":
            # The queries are PARAMETERS, so their count is fixed at
            # construction -- a per-call override cannot be honoured the way it
            # can for pooling, and silently ignoring it would give the wrong
            # token count for a camera that asked for a denser grid.
            if out_tokens is not None and out_tokens != self.out_tokens:
                raise ValueError(
                    f"attention pooling was built with out_tokens={self.out_tokens}; "
                    f"a per-call override to {out_tokens} would need its own query set. "
                    f"Use pool='avg' for per-camera token counts.")
            feat = self.pool(feat)                      # (B, out_tokens, 256)
        else:
            if out_tokens is None:
                feat = self.pool(feat)                  # (B, 256, token_side, token_side)
            else:
                side = int(out_tokens ** 0.5)
                assert side * side == out_tokens, "out_tokens must be a perfect square"
                feat = F.adaptive_avg_pool2d(feat, (side, side))
            feat = feat.flatten(2).transpose(1, 2)      # (B, out_tokens, 256)
        return self.norm(self.proj(feat))               # (B, out_tokens, out_dim)
