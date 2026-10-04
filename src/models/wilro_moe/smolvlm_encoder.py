"""SmolVLM2 encoder half, shared by `wilro` and `wilro_moe`.

Everything from raw camera frames and a task string to (a) the VLM text stack's
per-layer K/V cache, (b) language embeddings, and (c) the DiT's vision tokens.
It knows nothing about what decodes them, which is the whole point: wilro runs a
single 16-layer DiT over this, wilro_moe runs a mixture of expert decoders, and
neither should own a second copy of SigLIP loading, LoRA injection, RoPE, the
ResNet path or the paraphrase draw.

Extracted 2026-09-05 verbatim from wilro_model.py. Attribute names are
unchanged, so every existing wilro checkpoint still loads bit-for-bit -- that is
asserted by a test, not assumed (587 state_dict keys, strict load from a
pre-refactor model, sample_actions max|diff| = 0.0).

The one thing the mixin does NOT provide is __init__: the two models build
different decoders, so each constructs its own VLM/LoRA/ResNet in the order it
wants and simply keeps the attribute names these methods read
(vision_model, connector, text_model, robot_visual_encoder, ...).
"""
import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .robot_visual_encoder import RobotVisualEncoder


class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen nn.Linear.

    W (frozen) + B @ A (trainable, rank r)
    Forward: x @ W^T + (x @ A^T) @ B^T * (alpha / r)
    """
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base  # frozen original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_dim = base.in_features
        out_dim = base.out_features

        # LoRA A: (rank, in_dim) — init with normal
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.02)
        # LoRA B: (out_dim, rank) — init with zero (so adapter starts as identity)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward (frozen)
        base_out = self.base(x)
        # LoRA: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def apply_lora_to_vision_layers(
    vision_model,
    num_layers: int,
    rank: int = 8,
    alpha: float = 16.0,
) -> int:
    """Apply LoRA adapters to the last `num_layers` of a SigLIP ViT encoder.

    Modifies vision_model.encoder.layers in-place, wrapping q_proj and v_proj
    with LoRALinear adapters. Base weights are frozen; LoRA params are trainable.

    Returns the number of trainable parameters added.
    """
    encoder_layers = vision_model.encoder.layers
    total_layers = len(encoder_layers)
    start_idx = max(0, total_layers - num_layers)

    trainable_params = 0
    for i in range(start_idx, total_layers):
        layer = encoder_layers[i]
        # Wrap q_proj and v_proj with LoRA
        for name in ["q_proj", "v_proj"]:
            original = getattr(layer.self_attn, name)
            if isinstance(original, LoRALinear):
                continue  # already wrapped
            lora = LoRALinear(original, rank=rank, alpha=alpha)
            setattr(layer.self_attn, name, lora)
            trainable_params += lora.lora_A.numel() + lora.lora_B.numel()

    return trainable_params


def apply_lora_to_text_layers(
    text_model,
    num_layers: int,
    rank: int = 8,
    alpha: float = 16.0,
) -> int:
    """Apply LoRA adapters to the last `num_layers` of a Llama-style text model.

    Modifies text_model.layers in-place, wrapping q_proj and v_proj with
    LoRALinear adapters. Base weights are frozen; LoRA params are trainable.

    This enables the text encoder to adapt to robot-specific instructions
    and spatial grounding while preserving the pretrained language model.

    Returns the number of trainable parameters added.
    """
    layers = text_model.layers
    total_layers = len(layers)
    start_idx = max(0, total_layers - num_layers)

    trainable_params = 0
    for i in range(start_idx, total_layers):
        layer = layers[i]
        # Wrap q_proj and v_proj with LoRA
        for name in ["q_proj", "v_proj"]:
            original = getattr(layer.self_attn, name)
            if isinstance(original, LoRALinear):
                continue  # already wrapped
            lora = LoRALinear(original, rank=rank, alpha=alpha)
            setattr(layer.self_attn, name, lora)
            trainable_params += lora.lora_A.numel() + lora.lora_B.numel()

    return trainable_params


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (flow matching)
# ---------------------------------------------------------------------------

def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension must be even, got {dimension}")
    device = time.device
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = (1.0 / period) * 2.0 * math.pi
    sin_input = scaling[None, :] * time[:, None].float()
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


# ---------------------------------------------------------------------------
# RoPE helpers — Llama-style 1D rotary positional embedding for the VLM.
# Used only inside the encoder pass; the DiT does not use RoPE (the VLM K
# already carries positional rotation, which is enough for cross-attention).
# ---------------------------------------------------------------------------

def _build_rope_cache(
    seq_len: int, head_dim: int, base: float, device, dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each (1, seq_len, head_dim), ready for broadcast."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)        # (L, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                       # (L, head_dim)
    return emb.cos().to(dtype).unsqueeze(0), emb.sin().to(dtype).unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: (B, H, L, D); cos, sin: (1, L, D)."""
    cos = cos.unsqueeze(1)   # (1, 1, L, D)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# adaLN-Zero modulation
# ---------------------------------------------------------------------------

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x: (B, L, D) — shift/scale: (B, D). Broadcasts over L."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _hard_negative_perm(
    descs: list[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a HARD-negative partner index for the contrastive language loss.

    For each sample i, pick the in-batch partner j (j != i, with a DIFFERENT
    instruction) that shares the most words with i — i.e. the most *confusable*
    negative available, not a random one. Random batch pairing almost never
    lands a same-template pair (e.g. two "put both ... in the basket" tasks that
    differ only in the object nouns), so the contrastive hinge gets satisfied by
    trivially-different instructions and never pressures fine-grained object
    grounding. Hard negatives put the gradient exactly where eval fails.

    Similarity is word-overlap (Jaccard). For LIBERO's templated strings the
    same-template tasks share the entire template and differ only in the object
    nouns, so the confusable minimal pair scores highest automatically; no
    object vocabulary or extra model is needed.

    Returns (perm, valid):
      - perm[i] = chosen partner index (perm[i]=i when no partner exists)
      - valid[i] = whether a different-instruction partner was found (False rows
        are skipped downstream via pair_diff)
    perm need NOT be a bijection — several samples may share the same hardest
    negative, which is fine for the gather-based shuffle. O(B^2) set ops on CPU;
    negligible next to the VLM forward.
    """
    B = len(descs)
    word_sets = [set(d.lower().split()) for d in descs]
    perm = list(range(B))
    valid = [False] * B
    for i in range(B):
        wi = word_sets[i]
        best_score, best = -1.0, []
        for j in range(B):
            if j == i or descs[j] == descs[i]:
                continue
            wj = word_sets[j]
            union = len(wi | wj)
            score = (len(wi & wj) / union) if union else 0.0
            if score > best_score + 1e-9:
                best_score, best = score, [j]
            elif score > best_score - 1e-9:
                best.append(j)
        if best:
            # Random pick among ties so the partner varies across steps.
            perm[i] = best[int(torch.randint(len(best), (1,)).item())]
            valid[i] = True
    return (
        torch.tensor(perm, device=device, dtype=torch.long),
        torch.tensor(valid, device=device, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# DiT layer: self-attn + cross-attn(to VLM KV) + FFN, modulated by adaLN-Zero
# ---------------------------------------------------------------------------


class SmolVLMEncoderMixin:
    """Camera frames + task string -> VLM KV cache, language, vision tokens."""

    # Non-parameter state these methods read. A host that builds the VLM but
    # forgets one of these fails at the FIRST forward with a bare AttributeError
    # from deep inside _encode_language, twenty minutes into a run -- so the
    # contract is a method rather than a comment, and both hosts call it.
    def init_encoder_state(self):
        self._lang_max_len = 48
        self._paraphrase_cache: dict = {}     # instruction -> surface variants
        self._paraphrase_table = None
        self._paraphrase_file = None          # lazily loaded --paraphrase_file
        self._paraphrase_announced = False
        self._last_vlm_hidden = None

    def _encode_images(
        self,
        batch: dict,
        B: int,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode images through VLM vision encoder.

        Args:
            batch: input batch with camera images
            B: batch size
            return_intermediate: if True, also return intermediate hidden states
                from the vision encoder (for VLM-vision Robot CA source)

        Returns:
            vis_tokens: (B, L_vis, hidden_size) — connector-projected vision tokens
                for VLM text input
            intermediate_features: (B, L_vis, hidden_size) — intermediate layer
                features from vision encoder (only if return_intermediate=True,
                else None). These are SigLIP features that are naturally
                language-vision aligned.
        """
        vlm_dtype = next(self.vision_model.parameters()).dtype
        layer_offset = getattr(self.config, "vlm_vision_layer_offset", -3)
        all_vis: list[torch.Tensor] = []
        all_intermediate: list[torch.Tensor] = []

        for cam_key in self.config.cameras_for_vision_state_concat:
            if cam_key not in batch:
                continue
            imgs = batch[cam_key]
            img = imgs[:, -1] if imgs.dim() == 5 else imgs
            img = img * 2.0 - 1.0

            target = self.config.vision_input_size
            h, w = img.shape[-2], img.shape[-1]
            if h != w:
                max_dim = max(h, w)
                pad = (
                    (max_dim - w) // 2, max_dim - w - (max_dim - w) // 2,
                    (max_dim - h) // 2, max_dim - h - (max_dim - h) // 2,
                )
                img = F.pad(img.float(), pad, value=-1.0)
            if img.shape[-2] != target or img.shape[-1] != target:
                img = F.interpolate(img.float(), size=(target, target),
                                     mode="bilinear", align_corners=False).to(vlm_dtype)
            else:
                img = img.to(vlm_dtype)

            # Forward through vision encoder
            if return_intermediate:
                vis_output = self.vision_model(
                    pixel_values=img, output_hidden_states=True
                )
                vis_hidden = vis_output.last_hidden_state
                # Extract intermediate layer features (before connector)
                # hidden_states[0] = embedding output, hidden_states[1..N] = layer outputs
                # layer_offset=-3 means third-to-last transformer layer
                intermediate = vis_output.hidden_states[layer_offset]
                # Project intermediate features through connector to match text dim
                # .contiguous() required: connector's pixel_shuffle uses .view() which
                # fails on non-contiguous tensors (hidden_states may have stride gaps)
                intermediate_proj = self.connector(intermediate.contiguous())
                all_intermediate.append(intermediate_proj)
            else:
                vis_hidden = self.vision_model(pixel_values=img).last_hidden_state

            vis_tokens = self.connector(vis_hidden)
            all_vis.append(vis_tokens)

        if not all_vis:
            device = batch["observation.state"].device
            empty = torch.zeros(B, 0, self.hidden_size, device=device, dtype=torch.bfloat16)
            return empty, None

        vis_tokens = torch.cat(all_vis, dim=1)
        intermediate_features = torch.cat(all_intermediate, dim=1) if all_intermediate else None
        return vis_tokens, intermediate_features

    def _sample_paraphrase(self, desc: str) -> str:
        """One surface variant of `desc`, drawn per sample per step.

        Built lazily and cached: LIBERO has ~40 unique instructions, so after a
        few hundred steps nothing new arrives. Sampling per CALL rather than
        per epoch is what stops surface form from being a usable key -- a fixed
        rewrite would just be a second table to memorise.

        Written variants only -- the built-in table, or a file that overrides
        it. paraphrase.py's template generator is for DRAFTING new entries and
        is deliberately not on this path: a template that mangles a sentence
        must not reach the model without a human having read it. An instruction
        in neither source trains unaugmented, which the trainer preflight
        refuses to start on.
        """
        key = " ".join(str(desc).split())
        variants = self._paraphrase_cache.get(key)
        if variants is None:
            from libero_paraphrase import load_table, table_variants
            if self._paraphrase_file is None:
                path = str(getattr(self.config, "paraphrase_file", "") or "")
                self._paraphrase_file = load_table(path) if path else {}
            variants = (self._paraphrase_file.get(key)
                        or table_variants(key) or [key])
            lim = int(getattr(self.config, "paraphrase_limit", 0) or 0)
            if lim and len(variants) > lim:
                variants = variants[:lim]
            self._paraphrase_cache[key] = variants
            if len(variants) == 1:
                print(f"[wilro] paraphrase: NO variants for {key!r} -- this "
                      f"instruction trains UNAUGMENTED while the rest vary")
            elif not self._paraphrase_announced:
                self._paraphrase_announced = True
                print(f"[wilro] paraphrase augmentation ON; first instruction "
                      f"gets {len(variants)} variants")
        if len(variants) == 1:
            return variants[0]
        return variants[int(torch.randint(len(variants), (1,)).item())]

    def _encode_language(self, batch: dict, device: torch.device) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        descs = batch.get("task_description")
        if not descs:
            descs = batch.get("task")
        if not descs or not any(descs):
            return None
        # Only the ENCODER sees the variant. compute_loss reads the canonical
        # strings straight off `batch` for the contrastive hinge and for
        # _hard_negative_perm, both of which decide "same instruction?" by
        # exact string equality -- paraphrasing before that point would make
        # two phrasings of one task read as two tasks. Eval always passes the
        # original string, and `self.training` keeps this off there.
        if self.training and getattr(self.config, "paraphrase_augment", False):
            if isinstance(descs, str):
                descs = self._sample_paraphrase(descs)
            else:
                descs = [self._sample_paraphrase(d) for d in descs]
        inputs = self.processor.tokenizer(
            descs, return_tensors="pt", padding=True, truncation=True,
            max_length=self._lang_max_len, add_special_tokens=True,
        )
        input_ids = inputs["input_ids"].to(device)
        lang_mask = inputs["attention_mask"].bool().to(device)
        lang_tokens = self.text_model.get_input_embeddings()(input_ids)
        return lang_tokens, lang_mask

    # =========================================================================
    # VLM encoder: run all layers, cache K/V from the trailing num_dit_layers
    # =========================================================================
    # NOTE: No @torch.no_grad() here — vision_model LoRA adapters need gradient
    # flow through: loss → DiT → vision_tokens → intermediate_features → connector
    # → vision_model (LoRA). The text_model portion runs under no_grad context
    # below since KV caches are detached and text weights are frozen.
    def _run_vlm_and_cache_kv(
        self, batch: dict,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, int, int,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
          kv_cache:        list of length num_dit_layers, each entry is (K, V)
                           with shape (B, num_kv_heads, L_vlm, head_dim).
                           K is post-RoPE rotation.
          vlm_kv_pad_mask: (B, L_vlm) bool — True at non-padded positions.
          L_vis:           number of vision tokens.
          L_lang:          number of language tokens.
          lang_embeddings: (B, L_lang, H) — VLM-processed language embeddings
                           from final hidden state (for DiT sequence injection).
          vlm_vision_features:  (B, L_vis, H) — intermediate vision features for
                           Robot CA from SigLIP ViT (with LoRA adaptation).
                           These features are naturally language-vision aligned
                           through SigLIP's contrastive pretraining.
        """
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device

        # Extract intermediate features for Robot CA (SigLIP ViT intermediate layers)
        # Under the ResNet source the VLM intermediate is never read, so do not
        # pay for output_hidden_states + a second connector pass to build it.
        # The VLM intermediate exists only to feed Robot CA, so it follows
        # use_vision_ca. The ResNet does not -- see __init__.
        need_intermediate = self.use_vision_ca and self.vision_token_source != "resnet"
        vis_tokens, intermediate_features = self._encode_images(batch, B, return_intermediate=need_intermediate)
        L_vis = vis_tokens.shape[1]

        # Vision token dropout (regularizer). Disabled in eval / sampling.
        vp = float(getattr(self.config, "vision_dropout_prob", 0.0)) if self.training else 0.0
        if self.training and L_vis > 0 and vp > 0.0:
            keep = torch.rand(B, L_vis, device=vis_tokens.device) > vp
            vis_tokens = vis_tokens * keep.unsqueeze(-1).to(vis_tokens.dtype)

        lang_result = self._encode_language(batch, device)
        if lang_result is not None:
            lang_tokens, lang_mask = lang_result
            lang_tokens = lang_tokens.to(vis_tokens.dtype)
            # Zero out pad slots so they contribute no signal to the VLM.
            lang_tokens = torch.where(
                lang_mask.unsqueeze(-1), lang_tokens, torch.zeros_like(lang_tokens),
            )
            L_lang = lang_tokens.shape[1]
        else:
            lang_tokens = None
            lang_mask = None
            L_lang = 0

        parts = [vis_tokens]
        if lang_tokens is not None:
            parts.append(lang_tokens)
        vlm_seq = torch.cat(parts, dim=1).to(torch.bfloat16)
        L_vlm = vlm_seq.shape[1]

        # Key-padding mask: vision always valid; language follows lang_mask.
        if lang_mask is not None:
            vis_mask = torch.ones(B, L_vis, device=device, dtype=torch.bool)
            vlm_kv_pad_mask = torch.cat([vis_mask, lang_mask], dim=1)
        else:
            vlm_kv_pad_mask = torch.ones(B, L_vlm, device=device, dtype=torch.bool)

        # Causal + key-padding mask for VLM self-attention. Vision and language
        # are concatenated, monotonically positioned; SmolVLM2 was pretrained
        # causal so we keep that.
        causal = torch.triu(
            torch.full((L_vlm, L_vlm), float("-inf"), device=device, dtype=vlm_seq.dtype),
            diagonal=1,
        )
        full_mask = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, L_vlm, L_vlm).clone()
        key_pad = ~vlm_kv_pad_mask
        full_mask.masked_fill_(key_pad.unsqueeze(1).unsqueeze(1), float("-inf"))

        # RoPE cache for the full VLM sequence (positions 0..L_vlm-1).
        cos, sin = _build_rope_cache(
            L_vlm, self.head_dim, self.rope_theta, device, vlm_seq.dtype,
        )

        # ── Text model forward ──────────────────────────────────────────────
        # KV cache is ALWAYS detached for numerical stability. The gradient path
        # through 16 DiT layers → cross-attn → KV cache → 8 text LoRA layers is
        # too deep and causes gradient explosion (NaN).
        #
        # Text LoRA still receives gradients through lang_embeddings (NOT detached
        # when text LoRA is enabled), which flows through DiT self-attention:
        #   loss → DiT → self-attn(lang_tokens) → lang_embeddings → text_model LoRA
        # This is a much shorter, more stable gradient path.
        text_lora_enabled = getattr(self.config, "text_lora_num_layers", 0) > 0
        text_ctx = nullcontext() if text_lora_enabled else torch.no_grad()

        with text_ctx:
            hidden = vlm_seq
            kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []

            for i, layer in enumerate(self.text_model.layers):
                residual = hidden
                h_in = layer.input_layernorm(hidden)

                Q = layer.self_attn.q_proj(h_in)
                K = layer.self_attn.k_proj(h_in)
                V = layer.self_attn.v_proj(h_in)

                Bn, Ln, _ = Q.shape
                Q = Q.view(Bn, Ln, self.num_heads, self.head_dim).transpose(1, 2)
                K = K.view(Bn, Ln, self.num_kv_heads, self.head_dim).transpose(1, 2)
                V = V.view(Bn, Ln, self.num_kv_heads, self.head_dim).transpose(1, 2)

                Q, K = _apply_rope(Q, K, cos, sin)

                # KV cache ALWAYS detached — cross-attn gradient path is too deep.
                if i in self._capture_set:
                    kv_cache.append((K.detach(), V.detach()))

                if self.num_kv_heads != self.num_heads:
                    r = self.num_heads // self.num_kv_heads
                    K_x = K.repeat_interleave(r, dim=1)
                    V_x = V.repeat_interleave(r, dim=1)
                else:
                    K_x, V_x = K, V

                attn = F.scaled_dot_product_attention(Q, K_x, V_x, attn_mask=full_mask, is_causal=False)
                attn = attn.transpose(1, 2).contiguous().view(Bn, Ln, self.num_heads * self.head_dim)
                attn = layer.self_attn.o_proj(attn)
                hidden = residual + attn

                residual = hidden
                h_in = layer.post_attention_layernorm(hidden)
                hidden = residual + layer.mlp(h_in)

            # Extract VLM-processed language embeddings from the final hidden state.
            # These are used as DiT sequence tokens so robot/action can self-attend
            # to language directly (language grounding for Robot CNN features).
            # When text LoRA is enabled, keep gradient flow through lang_embeddings
            # (shorter, more stable path via DiT self-attention).
            # When text LoRA is disabled, detach to save memory.
            lang_embeddings = None
            if L_lang > 0:
                if text_lora_enabled:
                    lang_embeddings = hidden[:, L_vis:L_vis + L_lang]  # gradient flows to text LoRA
                else:
                    lang_embeddings = hidden[:, L_vis:L_vis + L_lang].detach()

        # Stashed, not returned: wilro_moe's router pools the FULL multimodal
        # hidden state (vision + language, already fused by the text stack's
        # causal attention), while wilro only ever wants the language slice. An
        # attribute keeps the return signature -- and every wilro call site --
        # untouched.
        self._last_vlm_hidden = hidden

        return kv_cache, vlm_kv_pad_mask, L_vis, L_lang, lang_embeddings, intermediate_features

    # =========================================================================
    # DiT-side helpers: robot CNN, latents, time, input assembly
    # =========================================================================
    def _resnet_tokens(self, batch: dict) -> Optional[torch.Tensor]:
        """Robot tokens from the trainable ResNet-18, one grid per camera.

        Layout per camera: [ pool_N(f_t) , gate * pool_M(f_t - f_{t-k}) ], the
        motion half present only when `resnet_motion_tokens > 0`. Both halves
        come from ONE shared backbone -- proj/norm are per-token, so the second
        grid costs no parameters, and the ImageNet stem stays intact (stacking
        the two frames into a 6-channel conv1 would destroy it).
        """
        enc = self.robot_visual_encoder
        if enc is None:
            return None
        stride = int(getattr(self.config, "resnet_motion_stride", 1) or 1)
        # Per-camera grid. The wrist view carries contact geometry and wants a
        # denser grid than the third-person view, which only supplies coarse
        # approach context. Same backbone, different pooling, no extra params.
        fine_cams = set(getattr(self.config, "resnet_fine_cameras", None) or [])
        fine_tok = int(getattr(self.config, "resnet_fine_tokens", 0) or 0)
        out: list[torch.Tensor] = []
        for cam_key in self.resnet_cameras:
            if cam_key not in batch:
                continue
            imgs = batch[cam_key]
            if imgs.dim() == 5:
                cur = imgs[:, -1]
                # The trainer requests exactly [-stride*dt, 0.0], so the older
                # frame is index 0. Anything else means the window and the
                # config disagree, and silently differencing the wrong pair
                # would look like a weak-but-present motion signal rather than
                # a bug.
                older = imgs[:, 0] if imgs.shape[1] >= 2 else None
            else:
                cur, older = imgs, None

            if self.resnet_motion_tokens > 0:
                if older is None:
                    raise ValueError(
                        f"resnet_motion_tokens={self.resnet_motion_tokens} needs two "
                        f"camera frames, but '{cam_key}' arrived with "
                        f"{tuple(imgs.shape)}. The trainer must request "
                        f"[-{stride}*frame_time, 0.0] for the cameras; check that "
                        f"--resnet_motion_tokens reached build_datasets too.")
                fm_cur = enc.trunk(cur.float())
                fm_old = enc.trunk(older.float())
                n_tok = fine_tok if (fine_tok > 0 and cam_key in fine_cams) else enc.out_tokens
                toks = enc.tokens_from_map(fm_cur, out_tokens=n_tok)
                mot = enc.tokens_from_map(fm_cur - fm_old,
                                          out_tokens=self.resnet_motion_tokens)
                mot = self.resnet_motion_gate.to(mot.dtype) * mot
                out.append(torch.cat([toks, mot], dim=1))
            else:
                n_tok = fine_tok if (fine_tok > 0 and cam_key in fine_cams) else None
                out.append(enc(cur.float(), out_tokens=n_tok))

        if not out:
            return None
        return torch.cat(out, dim=1)

    def _compute_vision_tokens(
        self,
        batch: dict,
        vlm_robot_features: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Compute robot visual tokens for Robot CA.

        Source is `config.vision_token_source`: either the VLM's own SigLIP ViT
        intermediate layer (frozen base + LoRA) or a separate trainable
        ResNet-18. Exactly one of them -- see the config for why this replaces
        rather than adds.

        Args:
            batch: input batch (ResNet source reads the camera tensors)
            vlm_robot_features: pre-extracted VLM vision intermediate features
                from SigLIP ViT (with LoRA adaptation); None under the ResNet
                source, where it is never computed.

        Returns:
            vision_tokens: (B, R, hidden_size) — robot visual tokens
        """
        if self.vision_token_source == "resnet":
            toks = self._resnet_tokens(batch)
            if toks is None:
                return None
        elif vlm_robot_features is None:
            return None
        else:
            toks = vlm_robot_features
        vp = float(getattr(self.config, "vision_dropout_prob", 0.0)) if self.training else 0.0
        if vp > 0:
            B, R, _ = toks.shape
            keep = torch.rand(B, R, device=toks.device) > vp
            toks = toks * keep.unsqueeze(-1).to(toks.dtype)
        return toks
