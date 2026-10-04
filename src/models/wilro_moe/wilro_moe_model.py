"""WILRO-MoE — SmolVLM2 encoder + a mixture of expert decoders.

wiltechs_moe with the backbone swapped from Qwen3-VL-4B to SmolVLM2-500M.

That swap is not a config change: wiltechs_moe reaches into `vlm.model.visual`
with `grid_thw`, `vlm.model.language_model`, and Qwen3-VL's 3D mRoPE, none of
which SmolVLM2 has. The SmolVLM2 encoder already exists though -- it is wilro's
-- so this model is wilro's encoder under wiltechs_moe's decoder, and the
encoder half is IMPORTED (SmolVLMEncoderMixin), not copied.

Nothing here imports from wiltechs_moe: that module pulls in
Qwen3VLForConditionalGeneration at import time, which is the dependency
cac2de6 had to cut out of the eval harness. The two decoder pieces it does need
(MoERouter, the QFormer) are reproduced below with their reasoning intact.

Differences from wiltechs_moe worth knowing:
  * 4 experts x 8 layers = 32, not 4 x 9 = 36. SmolVLM2 has 32 text layers and
    the experts' KV bands are disjoint, so 36 does not fit.
  * `dit_hidden_size` defaults to 960 (== the VLM's hidden), so expert
    self-attention inherits 15/5/64 and no GQA re-derivation is needed. That
    also lets the experts reuse wilro's own DiTLayer verbatim.
  * No Vision CA sublayer, matching wiltechs_moe: the vision tokens (SigLIP
    intermediate or ResNet) sit IN the sequence and are reached by causal
    self-attention alone.
"""
import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from .wilro_moe_config import WilroMoEConfig
from .layers import RMSNorm, SwiGLU
from .robot_visual_encoder import RobotVisualEncoder
from .dit import DiTLayer
from .smolvlm_encoder import (
    SmolVLMEncoderMixin, apply_lora_to_vision_layers, apply_lora_to_text_layers,
    create_sinusoidal_pos_embedding, _build_rope_cache, _hard_negative_perm,
)


class MoERouter(nn.Module):
    """Pooled multimodal context + state + time + action -> expert weights."""

    def __init__(self, hidden_size, num_experts, vlm_hidden_size,
                 temperature=1.0, top_k=0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.top_k = top_k
        self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)
        self.router = nn.Sequential(
            nn.Linear(4 * hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, num_experts))
        # Small random init, NOT zeros. Zero init makes every logit identical at
        # step 0; any tiny data gradient then tips one expert ahead and the
        # softmax positive-feedback loop collapses to it within ~100 steps
        # (observed on the sibling: E3 at 100% by step 200).
        nn.init.normal_(self.router[-1].weight, std=0.02)
        nn.init.normal_(self.router[-1].bias, std=0.02)
        self._last_clean_weights = None

    def forward(self, state_emb, vlm_semantic_emb, time_emb, action_emb):
        B = state_emb.shape[0]
        vlm_proj = self.vlm_proj(vlm_semantic_emb)
        action_pool = action_emb.mean(dim=1)
        # POOL, do not squeeze. squeeze(1) is a no-op once there is more than one
        # state frame, and the cat below then mixes a 3-D tensor with three 2-D
        # ones -- "Tensors must have same number of dimensions: got 3 and 2".
        # --use_state_history with n_obs_steps > 1 crashed here, at the first
        # forward, for every n_obs_steps except 1. Flattening instead of pooling
        # would not fix it either: the router's first Linear is 4*hidden wide, so
        # n_obs frames would need 4+n_obs-1 slots. Mean-pooling keeps the width
        # fixed for any n_obs and matches how action_emb is reduced one line up.
        state_flat = state_emb.mean(dim=1) if state_emb.dim() == 3 else state_emb
        logits = self.router(torch.cat(
            [state_flat, vlm_proj, time_emb, action_pool], dim=-1)
        ) / max(self.temperature, 1e-6)
        # Diagnostics read the PRE-noise weights: those are what inference uses,
        # and the exploration noise below inflates peakedness badly at these
        # logit scales -- a router with no input dependence still reports
        # max_w 0.39 once N(0, 0.5) is added, not the 0.25 "uniform" suggests.
        self._last_clean_weights = F.softmax(logits, dim=-1).detach()
        if self.training:
            # Fixed 0.5, not scaled to the logit magnitude, so it keeps feeding
            # dead experts exploration signal instead of washing out as the
            # router grows confident.
            logits = logits + torch.randn_like(logits) * 0.5
        if 0 < self.top_k < self.num_experts:
            _, topk_idx = logits.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
            w = F.softmax(logits, dim=-1) * mask
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            w = F.softmax(logits, dim=-1)
        return w, w.mean(dim=0)


class ExpertDecoder(nn.Module):
    """One expert: `num_layers` DiT layers over its own band of VLM KV."""

    def __init__(self, hidden_size, num_layers, num_heads, num_kv_heads,
                 head_dim, intermediate_size, rms_norm_eps=1e-5, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DiTLayer(hidden_size=hidden_size, num_heads=num_heads,
                     num_kv_heads=num_kv_heads, head_dim=head_dim,
                     intermediate_size=intermediate_size,
                     rms_norm_eps=rms_norm_eps, dropout=dropout,
                     use_vision_ca=False)
            for _ in range(num_layers)])

    def forward(self, x, t_emb, expert_kv_cache, vlm_kv_pad_mask, self_attn_mask):
        for i, layer in enumerate(self.layers):
            vlm_k, vlm_v = expert_kv_cache[i % len(expert_kv_cache)]
            x = layer(x, t_emb=t_emb, vlm_k=vlm_k, vlm_v=vlm_v,
                      vlm_kv_pad_mask=vlm_kv_pad_mask,
                      self_attn_mask=self_attn_mask)
        return x


class WilroMoETransformer(SmolVLMEncoderMixin, nn.Module):
    """SmolVLM2 encoder (shared with wilro) + a mixture of expert decoders."""

    VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def __init__(self, config: WilroMoEConfig):
        super().__init__()
        self.config = config
        self.init_encoder_state()

        # ---- 1. SmolVLM2, frozen, all layers ----------------------------
        print(f"Loading {self.VLM_MODEL_ID} ...")
        vlm = AutoModelForImageTextToText.from_pretrained(
            self.VLM_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.processor = AutoProcessor.from_pretrained(self.VLM_MODEL_ID)
        vlm_model = vlm.model
        self.vision_model = vlm_model.vision_model
        self.connector = vlm_model.connector
        self.text_model = vlm_model.text_model
        self.num_vlm_layers = len(self.text_model.layers)

        tc = self.text_model.config
        self.hidden_size = int(tc.hidden_size)
        self.num_heads = int(tc.num_attention_heads)
        self.num_kv_heads = int(getattr(tc, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(tc, "head_dim", None) or (self.hidden_size // self.num_heads))
        self.intermediate_size = int(tc.intermediate_size)
        self.rms_norm_eps = float(getattr(tc, "rms_norm_eps", 1e-5))
        self.rope_theta = float(getattr(tc, "rope_theta", 10000.0))
        print(f"VLM: {self.num_vlm_layers} layers  hidden={self.hidden_size}  "
              f"heads={self.num_heads}  kv_heads={self.num_kv_heads}  "
              f"head_dim={self.head_dim}  intermediate={self.intermediate_size}")
        if config.d_model != self.hidden_size:
            config.d_model = self.hidden_size
        for c in (self.vision_model, self.connector, self.text_model):
            for p in c.parameters():
                p.requires_grad = False
            c.eval()
        del vlm

        n_vis_lora = int(getattr(config, "vision_lora_num_layers", 0) or 0)
        if n_vis_lora > 0:
            n = apply_lora_to_vision_layers(
                self.vision_model, num_layers=n_vis_lora,
                rank=config.lora_rank, alpha=config.lora_alpha)
            print(f"[wilro_moe] SigLIP ViT LoRA: {n_vis_lora} layers, "
                  f"rank={config.lora_rank}, {n:,} trainable params")
            if config.vision_token_source == "resnet":
                print("  [WARN] vision_token_source='resnet' severs the only gradient "
                      "path to these adapters (the text stack runs under no_grad and "
                      "the KV cache is detached), so they will NOT train. Pass "
                      "--vision_lora_num_layers 0 to say so in the config.")
        n_txt_lora = int(getattr(config, "text_lora_num_layers", 0) or 0)
        if n_txt_lora > 0:
            apply_lora_to_text_layers(self.text_model, num_layers=n_txt_lora,
                                      rank=config.lora_rank, alpha=config.lora_alpha)

        # ---- 2. Which VLM layers each expert reads -----------------------
        # Disjoint contiguous bands, shallow experts on shallow layers. Capturing
        # ALL layers is what makes kv_cache[i] mean "VLM layer i" below; the
        # mixin appends in layer order, so a partial capture would silently
        # renumber the bands.
        n_exp = int(config.num_experts)
        depth = int(config.expert_num_layers)
        if config.vlm_capture_layers:
            capture = sorted(int(i) for i in config.vlm_capture_layers)
        else:
            need = n_exp * depth
            if need > self.num_vlm_layers:
                raise ValueError(
                    f"num_experts({n_exp}) x expert_num_layers({depth}) = {need} "
                    f"exceeds the VLM's {self.num_vlm_layers} text layers, and the "
                    f"experts' KV bands are disjoint. SmolVLM2-500M has 32 where "
                    f"Qwen3-VL-4B has 36, so wiltechs_moe's 4 x 9 does not port "
                    f"unchanged -- 4 x 8 fits exactly.")
            capture = (list(range(self.num_vlm_layers)) if need == self.num_vlm_layers
                       else torch.linspace(0, self.num_vlm_layers - 1, need).round().int().tolist())
        if len(capture) % n_exp != 0:
            raise ValueError(f"{len(capture)} capture layers is not divisible by "
                             f"num_experts={n_exp}")
        self.capture_indices = capture
        self._capture_set = set(capture)
        per = len(capture) // n_exp
        self.expert_kv_blocks = [capture[e * per:(e + 1) * per] for e in range(n_exp)]
        print(f"[wilro_moe] {n_exp} experts x {depth} layers over VLM layers {capture}")
        for e, blk in enumerate(self.expert_kv_blocks):
            print(f"  Expert {e}: VLM layers {blk}")

        # ---- 3. Experts and router ---------------------------------------
        self.dit_hidden = int(getattr(config, "dit_hidden_size", 0)) or self.hidden_size
        if self.dit_hidden % self.head_dim != 0:
            raise ValueError(f"dit_hidden_size ({self.dit_hidden}) must be divisible "
                             f"by head_dim ({self.head_dim})")
        if self.dit_hidden == self.hidden_size:
            sa_nh, sa_nkv = self.num_heads, self.num_kv_heads
        else:
            sa_nh = self.dit_hidden // self.head_dim
            gqa = max(1, self.num_heads // max(1, self.num_kv_heads))
            sa_nkv = max(1, sa_nh // gqa)
            while sa_nh % sa_nkv != 0:
                sa_nkv -= 1
        # DiTLayer's cross-attention consumes the VLM KV directly, so its head
        # geometry is the VLM's. At dit_hidden == hidden they coincide, which is
        # why wilro's DiTLayer can be reused unchanged.
        if self.dit_hidden != self.hidden_size:
            raise NotImplementedError(
                f"dit_hidden_size {self.dit_hidden} != VLM hidden {self.hidden_size}. "
                f"wilro's DiTLayer uses one head geometry for both self- and "
                f"cross-attention; a narrower expert needs the split sa_/ca_ "
                f"variant from wiltechs_vla, which is not ported here.")
        self.experts = nn.ModuleList([
            ExpertDecoder(hidden_size=self.dit_hidden, num_layers=depth,
                          num_heads=sa_nh, num_kv_heads=sa_nkv,
                          head_dim=self.head_dim,
                          intermediate_size=self.intermediate_size,
                          rms_norm_eps=self.rms_norm_eps, dropout=config.dropout)
            for _ in range(n_exp)])
        self.router = MoERouter(hidden_size=self.dit_hidden, num_experts=n_exp,
                                vlm_hidden_size=self.hidden_size,
                                temperature=float(config.router_temperature),
                                top_k=int(config.router_top_k))

        # ---- 4. Sequence embeddings --------------------------------------
        h = self.dit_hidden
        self.sink_token = nn.Parameter(torch.zeros(1, 1, h))
        nn.init.normal_(self.sink_token, std=0.02)
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, h), RMSNorm(h, eps=self.rms_norm_eps))
        self.action_in_proj = nn.Linear(config.action_dim, h)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, config.horizon, h))
        nn.init.normal_(self.action_pos_emb, std=0.02)
        self.final_norm = RMSNorm(h, eps=self.rms_norm_eps)
        self.action_out_proj = nn.Linear(h, config.action_dim)
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)
        self.time_embedder = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))

        # ---- 5. Vision tokens (shared with wilro) ------------------------
        # No Vision CA sublayer: like wiltechs_moe, these tokens live in the
        # SEQUENCE and are reached by causal self-attention only.
        self.use_vision_ca = False
        self.vision_token_source = getattr(config, "vision_token_source", "vlm")
        self.vlm_vision_layer_offset = getattr(config, "vlm_vision_layer_offset", -3)
        self.robot_visual_encoder = None      # state_dict name, kept from 2026-06
        self.resnet_motion_gate = None
        self.resnet_motion_tokens = 0
        self.resnet_cameras = []
        if self.vision_token_source == "resnet":
            self.robot_visual_encoder = RobotVisualEncoder(
                input_size=int(config.resnet_input_size),
                out_tokens=int(config.resnet_tokens),
                out_dim=h, pool=str(config.resnet_pool))
            self.resnet_motion_tokens = int(getattr(config, "resnet_motion_tokens", 0) or 0)
            if self.resnet_motion_tokens > 0:
                self.resnet_motion_gate = nn.Parameter(torch.zeros(1))
            self.resnet_cameras = (list(config.resnet_cameras)
                                   or list(config.cameras_for_vision_state_concat))
            print(f"[wilro_moe] vision tokens: ResNet-18, {config.resnet_tokens} tok "
                  f"x {len(self.resnet_cameras)} cam @ {config.resnet_input_size}px")
        else:
            print(f"[wilro_moe] vision tokens: SigLIP intermediate layer "
                  f"{self.vlm_vision_layer_offset}")

        # Per-expert read of the shared vision tokens. Zero-init output over a
        # residual, so an adapter that never trains is the identity map rather
        # than noise -- which is what an expert the router starved would
        # otherwise wake up to.
        adim = int(getattr(config, "resnet_expert_adapter_dim", 0) or 0)
        self.expert_vision_adapters = None
        self.expert_vision_gates = None
        if adim > 0:
            self.expert_vision_adapters = nn.ModuleList()
            for _ in range(n_exp):
                mlp = nn.Sequential(RMSNorm(h, eps=self.rms_norm_eps),
                                    nn.Linear(h, adim), nn.SiLU(),
                                    nn.Linear(adim, h))
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)
                self.expert_vision_adapters.append(mlp)
            self.expert_vision_gates = nn.Parameter(torch.zeros(n_exp))
            per = (h * adim + adim * h + adim + h) / 1e6
            print(f"[wilro_moe] per-expert vision adapters: dim {adim}, "
                  f"{per:.2f}M x {n_exp} = {per * n_exp:.2f}M, zero-init residual")

        self.use_state_history = bool(getattr(config, "use_state_history", False))
        self.num_latent_tokens = 0            # wilro's latent path is unused here
        self.gradient_checkpointing = False
        self._rope_cache = None
        self._last_router_usage = None
        self._last_router_max_w = None
        self._last_router_entropy = None
        self._last_expert_disagreement = None
        self._last_expert_ambiguity = None
        # Opt-in per-denoising-step routing trace. The router runs INSIDE
        # _run_dit, which sample_actions calls num_inference_steps times, and
        # two of its four inputs (time_emb, and the pooled NOISY action) change
        # every step -- so the expert mixture is re-decided at every step of
        # the ODE, not once per chunk. Nothing recorded that until now.
        self._record_routing = False
        self._routing_trace = None
        self._last_loss_components = None
        self._capture_attention_stats = False
        self._last_attention_stats = None
        self._last_cross_attention_stats = None
        self._last_vlm_hidden = None

    def sample_noise(self, shape, device):
        rho = self.config.noise_temporal_correlation
        noise = torch.randn(shape, device=device)
        if rho == 0.0 or shape[1] == 1:
            return noise
        scale = math.sqrt(1.0 - rho * rho)
        for t in range(1, shape[1]):
            noise[:, t] = rho * noise[:, t - 1] + scale * noise[:, t]
        return noise

    def sample_time(self, B, device):
        # "lognormal": SD3-style logit-normal t = sigmoid(N(mean, std)). A negative
        # mean biases toward LOW t (= x_t≈actions), spending more capacity on the
        # fine-detail denoising that sets placement precision. Default "uniform".
        if getattr(self.config, "time_sampling", "uniform") == "lognormal":
            z = (torch.randn(B, device=device) * self.config.time_lognormal_std
                 + self.config.time_lognormal_mean)
            return torch.sigmoid(z).clamp(0.001, 0.999)
        t = torch.rand(B, device=device)
        return t * 0.998 + 0.001

    def compute_loss(self, batch: dict) -> torch.Tensor:
        actions = batch["action"].float().nan_to_num(0.0).clamp(-10.0, 10.0)
        B = actions.shape[0]
        device = actions.device

        # ── Encoder: run VLM once, cache KV + extract lang + robot features ──
        (kv_cache, vlm_kv_pad_mask, L_vis, L_lang,
         lang_embeddings, vlm_vision_features) = self._run_vlm_and_cache_kv(batch)

        # ── DiT-side conditioning that does NOT depend on noise ─────
        vision_tokens = self._compute_vision_tokens(batch, vlm_vision_features)
        latents = self._generate_latents(batch, B, device, torch.bfloat16)

        # ── Action prefix for async execution training ──────────────
        action_prefix = None
        max_prefix = int(getattr(self.config, "max_action_prefix_steps", 0))
        if self.training and max_prefix > 0:
            prefix_len = torch.randint(0, max_prefix + 1, (1,), device=device).item()
            if prefix_len > 0:
                action_prefix = actions[:, :prefix_len]

        # ── Flow matching: build noisy actions, predict velocity ────
        noise = self.sample_noise(actions.shape, device)
        t = self.sample_time(B, device)
        t_exp = t[:, None, None]
        x_t = t_exp * noise + (1.0 - t_exp) * actions
        u_t = noise - actions

        v_t = self._run_dit(
            batch, x_t.to(torch.bfloat16), t, kv_cache, vlm_kv_pad_mask,
            vision_tokens, latents, action_prefix, lang_embeddings,
            L_vis=L_vis, L_lang=L_lang,
        ).float()

        # ── Per-position / per-dim weighting ────────────────────────
        loss = F.mse_loss(v_t, u_t, reduction="none")
        if self.config.action_dim_weights:
            dim_w = torch.tensor(self.config.action_dim_weights, device=loss.device, dtype=loss.dtype)
            loss = loss * dim_w[None, None, :]

        # Kept before pos_w is applied: the per-position diagnostic wants the RAW
        # flow loss, not the one already reweighted by the thing being questioned.
        loss_raw = loss
        H = loss.shape[1]
        n_exec = self.config.n_action_steps
        pos_w = torch.ones(H, device=loss.device, dtype=loss.dtype)
        pos_w[n_exec:] = self.config.future_steps_weight
        if self.config.pos_decay_lambda > 0.0:
            pos = torch.arange(H, device=loss.device, dtype=loss.dtype)
            pos_w = pos_w * torch.exp(-self.config.pos_decay_lambda * pos)
        # Combined per-(B, H, 1) weight: chunk-position weight × optional gripper
        # phase weight. Built once and reused in the denominator so the loss stays
        # a weighted MEAN (reweight, not rescale) — effective LR is unchanged.
        w_pos = pos_w[None, :, None].expand(loss.shape[0], H, 1).clone()

        # AWR: a per-SAMPLE weight, exp(A/beta), supplied by the trainer from a
        # collected corpus. Folded into w_pos so it lands in the DENOMINATOR too
        # and the loss stays a weighted MEAN -- reweight, not rescale, so the
        # effective LR does not move with beta. Absent => unchanged.
        awr_w = batch.get("awr_weight")
        if awr_w is not None:
            w_pos = w_pos * awr_w.to(w_pos.device, w_pos.dtype).view(-1, 1, 1)

        gpw = float(getattr(self.config, "gripper_phase_weight", 1.0))
        if gpw != 1.0:
            # Up-weight frames near a gripper open<->close transition (grasp /
            # release): |Δgripper| over the chunk, dilated to a ±window, scaled to
            # gpw. Off-window frames keep weight 1.0.
            gidx = getattr(self.config, "gripper_action_index", -1)
            g = actions[:, :, gidx]                                      # (B, H)
            dg = torch.zeros_like(g)
            dg[:, 1:] = (g[:, 1:] - g[:, :-1]).abs()
            trans = (dg > self.config.gripper_transition_thresh).to(loss.dtype)
            win = int(getattr(self.config, "gripper_transition_window", 2))
            if win > 0:
                trans = F.max_pool1d(trans.unsqueeze(1), 2 * win + 1,
                                     stride=1, padding=win).squeeze(1)
            phase_w = 1.0 + (gpw - 1.0) * trans                          # (B, H)
            w_pos = w_pos * phase_w[:, :, None]
        loss = loss * w_pos

        loss_dtype = loss.dtype
        Bn, Hn, Dn = loss.shape

        is_pad = batch.get("action_is_pad", batch.get("actions_id_pad"))
        valid_t = (~is_pad.bool()).to(loss_dtype) if is_pad is not None \
                  else torch.ones(Bn, Hn, device=loss.device, dtype=loss_dtype)

        dim_pad = batch.get("action_dim_pad")
        valid_d = (~dim_pad.bool()).to(loss_dtype) if dim_pad is not None \
                  else torch.ones(Bn, Dn, device=loss.device, dtype=loss_dtype)

        valid_cells = valid_t.unsqueeze(-1) * valid_d.unsqueeze(1)
        loss = loss * valid_cells
        denom = (w_pos * valid_cells).sum().clamp(min=1e-6)
        main_loss = loss.sum() / denom

        # Per-horizon-position flow loss, unweighted and pad-masked. The headline
        # flow number averages all H positions, but only n_action_steps of them are
        # ever executed -- so it cannot say whether the error lives in the part that
        # runs or in the far horizon, which is intrinsically less predictable.
        # E[u^2] is recorded alongside because it varies with position too: the
        # scale-free reading is loss / E[u^2], where 1.0 means "predicted nothing".
        if getattr(self, "_record_position_loss", False):
            with torch.no_grad():
                amb = getattr(self, "_position_ambiguity", None)
                amb_sum = (((amb.float() * valid_cells).sum(dim=(0, 2)).cpu())
                           if amb is not None
                           else torch.zeros(Hn, dtype=torch.float32))
                self._position_loss = (
                    (loss_raw * valid_cells).sum(dim=(0, 2)).detach().float().cpu(),
                    ((u_t ** 2) * valid_cells).sum(dim=(0, 2)).detach().float().cpu(),
                    valid_cells.sum(dim=(0, 2)).detach().float().cpu(),
                    amb_sum,
                )
                # Same quantities kept PER CELL (B, H) rather than summed over the
                # batch. The horizon profile only needs per-position sums, but
                # bucketing by "distance to the nearest gripper transition" is a
                # per-SAMPLE alignment -- the transition sits at a different
                # position in every chunk -- so the caller must do the summing
                # after it has located each row's transitions.
                self._cell_loss = (
                    (loss_raw * valid_cells).sum(dim=2).detach().float().cpu(),
                    ((u_t ** 2) * valid_cells).sum(dim=2).detach().float().cpu(),
                    valid_cells.sum(dim=2).detach().float().cpu(),
                    ((amb.float() * valid_cells).sum(dim=2).detach().float().cpu()
                     if amb is not None else torch.zeros(Bn, Hn)),
                )

        # ── Contrastive language loss: permute the LANGUAGE portion of
        # the cached KV across batch and re-run only the DiT. Avoids a
        # second full VLM forward.
        contrastive_w = float(getattr(self.config, "contrastive_loss_weight", 0.0))
        contrastive_v = 0.0
        # The hinge alone cannot be read. relu(margin - diff_sq) = 0 says only
        # "satisfied", and 0.051 against 5.0 are the same 0.0000 while meaning
        # opposite things about whether hard negatives can bite. Record the
        # separation itself.
        contr_stats: dict[str, float] = {}
        if (
            self.training and contrastive_w > 0.0
            and L_lang > 0 and B >= 2
        ):
            # Prefer task_description (may be rewritten for spatial grounding)
            # over raw task string from dataset.
            descs = batch.get("task_description") or batch.get("task")
            use_hard_neg = getattr(self.config, "contrastive_hard_negatives", False)

            if (
                use_hard_neg
                and descs is not None and len(descs) == B
            ):
                perm, pair_diff = _hard_negative_perm(descs, device)
            else:
                perm = torch.randperm(B, device=device)
                if (perm == torch.arange(B, device=device)).any():
                    perm = torch.roll(perm, shifts=1, dims=0)

                if descs is not None and len(descs) == B:
                    perm_cpu = perm.detach().cpu().tolist()
                    pair_diff = torch.tensor(
                        [descs[i] != descs[perm_cpu[i]] for i in range(B)],
                        device=device, dtype=torch.bool,
                    )
                else:
                    pair_diff = torch.ones(B, device=device, dtype=torch.bool)

            if pair_diff.any():
                # Treat the wrong-language prediction as a FIXED negative
                # target (stop-gradient). Building the shuffled KV cache and
                # running the second DiT forward under no_grad frees the
                # activations immediately instead of storing a full 309M-param
                # backward graph — this removes the ~2x memory blow-up. The
                # contrastive gradient still flows through v_t, pushing the
                # correct-language prediction away from the (detached) wrong one.
                with torch.no_grad():
                    shuffled_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
                    for K, V in kv_cache:
                        K_shuf = K.clone()
                        V_shuf = V.clone()
                        K_shuf[:, :, L_vis:L_vis + L_lang, :] = K[perm, :, L_vis:L_vis + L_lang, :]
                        V_shuf[:, :, L_vis:L_vis + L_lang, :] = V[perm, :, L_vis:L_vis + L_lang, :]
                        shuffled_cache.append((K_shuf, V_shuf))
                    shuffled_pad_mask = vlm_kv_pad_mask.clone()
                    shuffled_pad_mask[:, L_vis:L_vis + L_lang] = vlm_kv_pad_mask[perm][:, L_vis:L_vis + L_lang]
                    # Shuffle lang_embeddings for the contrastive forward
                    shuffled_lang = lang_embeddings[perm] if lang_embeddings is not None else None

                    v_wrong = self._run_dit(
                        batch, x_t.to(torch.bfloat16), t,
                        shuffled_cache, shuffled_pad_mask,
                        vision_tokens, latents, action_prefix, shuffled_lang,
                        L_vis=L_vis, L_lang=L_lang, record=False,
                    ).float()

                diff_sq = (v_t - v_wrong).pow(2).mean(dim=[1, 2])
                margin = float(getattr(self.config, "contrastive_margin", 0.05))
                with torch.no_grad():
                    ds = diff_sq[pair_diff].detach().float()
                    if ds.numel():
                        contr_stats = {
                            "diff_sq_mean": float(ds.mean()),
                            "diff_sq_min": float(ds.min()),
                            "diff_sq_p10": float(ds.quantile(0.1)),
                            "diff_sq_under": float((ds < margin).float().mean()),
                            "n_pairs": float(ds.numel()),
                            "hard_neg": float(bool(use_hard_neg)),
                        }
                hinge = F.relu(margin - diff_sq) * pair_diff.float()
                n_valid = pair_diff.float().sum().clamp(min=1.0)
                loss_contrastive = hinge.sum() / n_valid
                contrastive_v = float(loss_contrastive.detach())
                main_loss = main_loss + contrastive_w * loss_contrastive

        self._last_loss_components = {
            "main": float(main_loss.detach() - contrastive_w * contrastive_v),
            "contrastive": contrastive_v,
            **contr_stats,
        }
        return main_loss

    def forward(self, batch: dict) -> tuple:
        if self.training:
            return self.compute_loss(batch), {}
        return self.sample_actions(batch), {}

    def flow_actions_from_noise(self, batch: dict, x_init: torch.Tensor) -> torch.Tensor:
        """Deterministic flow ODE solution from a GIVEN initial noise x_init
        (B, horizon, action_dim), returned for the FULL horizon in normalized
        action space. Differentiable through the DiT / robot CNN / latent
        generator (the frozen VLM encoder still runs under no_grad inside
        _run_vlm_and_cache_kv).

        Used by RL (GRPO): the policy is N(flow_actions_from_noise(s, x1), sigma^2)
        conditioned on the stored noise latent x1, so action log-probs are exact
        and importance ratios are computable. Mirrors sample_actions' integration
        exactly.

        NOTE: Wraps the DiT loop in autocast so the DiT linear layers (stored in
        fp32) accept bf16 noisy_actions without dtype mismatch. Caller controls
        grad/no_grad context.
        """
        B = x_init.shape[0]
        device = x_init.device

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext()
        )

        with autocast_ctx:
            (kv_cache, vlm_kv_pad_mask, L_vis, L_lang,
             lang_embeddings, vlm_vision_features) = self._run_vlm_and_cache_kv(batch)
            vision_tokens = self._compute_vision_tokens(batch, vlm_vision_features)
            latents = self._generate_latents(batch, B, device, torch.bfloat16)

            N = int(getattr(self.config, "num_inference_steps", 10))
            x_t = x_init.float()
            dt = -1.0 / N
            t = torch.ones(B, device=device, dtype=torch.float32)
            for _ in range(N):
                v_t = self._run_dit(
                    batch, x_t.to(torch.bfloat16), t, kv_cache, vlm_kv_pad_mask,
                    vision_tokens, latents, action_prefix=None, lang_tokens=lang_embeddings,
                    L_vis=L_vis, L_lang=L_lang,
                ).float()
                x_t = x_t + dt * v_t
                t = t + dt
        return x_t

    @torch.no_grad()
    def sample_actions(self, batch: dict, full: bool = False) -> torch.Tensor:
        B = batch["observation.state"].shape[0]
        device = batch["observation.state"].device
        if self._record_routing:
            self._routing_trace = []      # one trace per chunk, N entries long

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext()
        )

        with autocast_ctx:
            (kv_cache, vlm_kv_pad_mask, L_vis, L_lang,
             lang_embeddings, vlm_vision_features) = self._run_vlm_and_cache_kv(batch)
            vision_tokens = self._compute_vision_tokens(batch, vlm_vision_features)
            latents = self._generate_latents(batch, B, device, torch.bfloat16)

            N = int(getattr(self.config, "num_inference_steps", 10))
            # A CYCLE OF m VECTORS, chunk k taking t[k % m]. Still fully
            # deterministic -- the same layout replays bit for bit -- but the
            # noise is no longer identical from one chunk to the next, which
            # is the one thing a single ticket cannot be.
            #
            # It exists because a single frozen vector costs long T0 about 50
            # points (13-15% against Gaussian's 60-73%) while costing the
            # short suites little, and the suites line up by chunks per
            # successful episode: spatial 29.0, goal 29.5, object 39.4 all
            # yield 20/20 tickets, long's 79.2 yields nothing. Training never
            # showed the model two consecutive chunks with the same x_1, so
            # whether the damage is the VECTOR or its repetition is a real
            # question, and m is the dial that answers it: m=1 is the ticket,
            # large m with random vectors is ordinary sampling.
            cyc = getattr(self, "_noise_cycle", None)
            ticket = getattr(self, "_noise_ticket", None)
            if cyc is not None:
                k = int(getattr(self, "_noise_cycle_k", 0))
                self._noise_cycle_k = k + 1
                # (m, H, D) is one cycle for every env. (B, m, H, D) is a
                # DIFFERENT cycle per env, which is what the search needs: one
                # batch scores n_par candidates against the same layout, and a
                # candidate is now an m-tuple rather than a vector.
                ticket = (cyc[:, k % cyc.shape[1]] if cyc.dim() == 4
                          else cyc[k % cyc.shape[0]])
            if ticket is not None:
                # The golden-ticket hypothesis (Patil et al. 2026): a frozen
                # generative policy can be improved by replacing x_1 ~ N(0,I)
                # with ONE well-chosen constant vector, reused at every action
                # step and every episode. Only sampling changes; compute_loss
                # still draws from the prior, which is what keeps this
                # applicable to an already-trained checkpoint.
                #
                # This is the EPISODE-level intervention. The 2026-09-23
                # best-of-N null established that per-chunk selection buys
                # nothing here -- four draws at one state are interchangeable
                # -- while a policy_seed change, which is exactly a change of
                # the whole episode's noise, flips outcomes. A constant ticket
                # acts at the level that was measured to matter.
                x_t = ticket.to(device=device, dtype=torch.float32)
                if x_t.dim() == 2:
                    x_t = x_t.unsqueeze(0).expand(B, -1, -1).contiguous()
                elif x_t.shape[0] != B:
                    raise ValueError(
                        f"_noise_ticket has batch {x_t.shape[0]} but the batch "
                        f"is {B}; a per-env ticket must name every env.")
            else:
                x_t = self.sample_noise(
                    (B, self.config.horizon, self.config.action_dim), device=device,
                )
            # Applied HERE and not inside sample_noise(): compute_loss calls the
            # same helper and training must keep x_1 ~ N(0, I).
            #
            # `_noise_scale_override` is a (B,) tensor set by the policy's stall
            # escape, so the envs that have stopped moving get more noise and the
            # healthy ones are untouched. A batch-wide scalar cannot do that.
            ov = getattr(self, "_noise_scale_override", None)
            ns = float(getattr(self.config, "sample_noise_scale", 1.0) or 1.0)
            if ticket is not None:
                # Scaling a ticket produces a DIFFERENT ticket, and the one
                # that was searched is the one that was scored. Leave it alone.
                ov, ns = None, 1.0
            if ov is not None:
                x_t = x_t * ov.to(x_t.device, x_t.dtype).view(-1, 1, 1)
            elif ns != 1.0:
                x_t = x_t * ns
            dt = -1.0 / N
            t = torch.ones(B, device=device, dtype=torch.float32)

            for _ in range(N):
                v_t = self._run_dit(
                    batch, x_t.to(torch.bfloat16), t, kv_cache, vlm_kv_pad_mask,
                    vision_tokens, latents, action_prefix=None, lang_tokens=lang_embeddings,
                    L_vis=L_vis, L_lang=L_lang,
                ).float()
                x_t = x_t + dt * v_t
                t = t + dt

        # Truncated by default, which is all the plain chunked path executes.
        # `full=True` returns every horizon step, which is what temporal
        # ensembling averages over -- the whole point is that 31 of the 32
        # chunks covering a timestep are otherwise thrown away here.
        return x_t if full else x_t[:, : self.config.n_action_steps]

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    # ---- decoder: mixture of experts --------------------------------------

    def _generate_latents(self, batch, B, device, dtype):
        """wilro's task-conditional latent path is unused here. The experts
        reach the instruction through their cross-attention to the VLM KV."""
        return None

    def _pool_vlm_semantic(self, vlm_hidden, pad_mask):
        """Mean-pool the VLM's final hidden state over valid tokens.

        Vision and language are already fused by the text stack's causal
        attention, so one pool of the final layer gives the router the whole
        multimodal context. Hidden states rather than the KV cache's V: hidden
        is always `hidden_size`, so there is no GQA head-count mismatch.
        """
        m = pad_mask.unsqueeze(-1).to(vlm_hidden.dtype)
        return (vlm_hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _build_expert_input(self, batch, noisy_actions, vision_tokens):
        """[sink, state, vision, action] -> (seq, action_start_idx, ...).

        Vision tokens go BEFORE the action tokens because this model has no
        Vision CA sublayer: causal self-attention over the sequence is the only
        path from an action query to them.

        wiltechs_moe additionally puts K "thought" tokens here, from a QFormer
        over the deepest VLM layer's KV. Dropped 2026-09-05: reported as not
        earning its keep there, and it cost 18.4M plus a sequence region. No
        wilro_moe checkpoint existed yet, so removing it was free -- which is
        the only reason it is a deletion rather than a default-off flag.
        """
        B, H, _ = noisy_actions.shape
        dtype = noisy_actions.dtype
        sink = self.sink_token.expand(B, -1, -1).to(dtype)
        state = batch["observation.state"].float()
        if state.dim() == 2:
            state = state.unsqueeze(1)
        state = state.nan_to_num(0.0).clamp(-10.0, 10.0)
        state_tok = self.state_encoder(state).to(dtype)
        if state_tok.shape[1] > 1 and not self.use_state_history:
            state_tok = state_tok[:, -1:]
        action_emb = (self.action_in_proj(noisy_actions)
                      + self.action_pos_emb[:, :H]).to(dtype)
        parts = [sink, state_tok]
        if vision_tokens is not None:
            parts.append(vision_tokens.to(dtype))
        vis_lo = 1 + state_tok.shape[1]
        vis_hi = vis_lo + (0 if vision_tokens is None else vision_tokens.shape[1])
        parts.append(action_emb)
        seq = torch.cat(parts, dim=1)
        return seq, seq.shape[1] - H, state_tok, action_emb, (vis_lo, vis_hi)

    def _run_dit(self, batch, noisy_actions, timesteps, kv_cache, vlm_kv_pad_mask,
                 vision_tokens, latents, action_prefix=None, lang_tokens=None,
                 L_vis=0, L_lang=0, record=True):
        """Same signature as wilro's `_run_dit`, so the loss and sampling code
        extracted from it works unchanged. `latents`, `action_prefix` and
        `lang_tokens` are accepted and ignored: this model has no latent path,
        no async prefix, and reaches language through the experts' cross-attn to
        the VLM KV rather than by injecting it into the sequence.
        """
        device, dtype = noisy_actions.device, noisy_actions.dtype
        t_emb = self.time_embedder(
            create_sinusoidal_pos_embedding(timesteps, self.dit_hidden).to(dtype).float()
        ).to(dtype)

        seq, action_start_idx, state_tok, action_emb, vis_span = self._build_expert_input(
            batch, noisy_actions, vision_tokens)
        L = seq.shape[1]
        causal = torch.triu(torch.full((L, L), float("-inf"), device=device,
                                       dtype=dtype), diagonal=1)

        hidden = self._last_vlm_hidden
        vlm_semantic = self._pool_vlm_semantic(hidden, vlm_kv_pad_mask).to(dtype)
        weights, usage = self.router(state_tok, vlm_semantic, t_emb, action_emb)
        if self._record_routing and record:
            # `and record` for the same reason every other statistic here carries
            # it: the contrastive negative calls _run_dit a second time with the
            # LANGUAGE band of the KV permuted. Without the guard the trace would
            # interleave real and permuted-instruction forwards while claiming to
            # describe the real one. Dropping that guard on _last_router_usage
            # once collapsed the router to a single expert by step 200.
            #
            # PRE-noise weights: what inference actually uses. Appended in ODE
            # order, so index 0 is t=1 (coarse) and index -1 is the last step
            # (fine placement) -- which is the axis the disjoint VLM bands were
            # meant to specialise along.
            cw = self.router._last_clean_weights
            if cw is not None:
                if self._routing_trace is None:
                    self._routing_trace = []
                self._routing_trace.append(cw.detach().float().cpu())
        # Per-sample stats alongside the batch mean. `usage` is a MEAN, so a
        # uniform CV^2 is ambiguous: every sample can be fully collapsed and
        # still average out flat if different samples collapse to different
        # experts. max_w and entropy read the PRE-noise weights, which is what
        # inference actually uses.
        if record:
            # record=False on the contrastive negative's forward. It runs AFTER
            # the real one, under torch.no_grad(), and without this guard it
            # OVERWRITES every router statistic -- so the log would describe
            # routing under PERMUTED instructions while claiming to describe the
            # real forward, and, far worse, `_last_router_usage` would be a
            # graph-detached tensor. The balance penalty reads it AFTER
            # compute_loss returns (it lives in the policy's forward), so a
            # detached value makes the penalty a constant: added to the loss,
            # contributing exactly zero gradient. Measured consequence of not
            # having this guard: router collapsed to one expert by step 200 with
            # `Router - Avg Abs Grad: 0.000000`, entropy 0.000, and nothing
            # pushing back.
            self._last_router_usage = usage
            cw = self.router._last_clean_weights
            if cw is not None:
                self._last_router_max_w = float(cw.max(dim=-1).values.mean())
                self._last_router_entropy = float(
                    -(cw.clamp(min=1e-9).log() * cw).sum(dim=-1).mean())

        lo, hi = vis_span
        outs = []
        for e, expert in enumerate(self.experts):
            expert_kv = [kv_cache[i] for i in self.expert_kv_blocks[e]]
            seq_e = seq
            if self.expert_vision_adapters is not None and hi > lo:
                vis = seq[:, lo:hi]
                delta = (self.expert_vision_gates[e].to(vis.dtype)
                         * self.expert_vision_adapters[e](vis))
                seq_e = torch.cat([seq[:, :lo], vis + delta, seq[:, hi:]], dim=1)
            # is_grad_enabled, NOT self.training: checkpointing exists to trade
            # compute for the backward graph, and whether a backward graph is
            # being built is exactly torch.is_grad_enabled(). Gating on
            # training broke RL, where the update must run in eval mode to
            # reproduce the rollout but still needs gradients.
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(
                    expert, seq_e, t_emb, expert_kv, vlm_kv_pad_mask, causal,
                    use_reentrant=False)
            else:
                x = expert(seq_e, t_emb=t_emb, expert_kv_cache=expert_kv,
                           vlm_kv_pad_mask=vlm_kv_pad_mask, self_attn_mask=causal)
            outs.append(self.action_out_proj(
                self.final_norm(x[:, action_start_idx:])))
        stacked = torch.stack(outs, dim=1)
        v = (weights.unsqueeze(-1).unsqueeze(-1) * stacked).sum(dim=1)

        # Do the experts DISAGREE, and is the disagreement WORTH anything?
        #
        # Two numbers, because the dimensionless one alone cannot answer the
        # question it looks like it answers.
        #
        # (1) AMBIGUITY -- read this one. Krogh-Vedelsby, exact per sample for
        #     any weights summing to 1:
        #
        #       (v_bar - u)^2 = sum_e w_e (v_e - u)^2  -  sum_e w_e (v_e-v_bar)^2
        #       ^ mixture MSE   ^ mean individual MSE     ^ AMBIGUITY
        #
        #     So the disagreement is the term the mixture SUBTRACTS: it is the
        #     ensemble gain itself, in loss units, and it needs no target to
        #     compute. Divided by the flow loss it says directly what fraction
        #     of the MSE the mixture is buying. No threshold required:
        #       ~0 of flow   the experts are redundant copies; 4x8 is computing
        #                    what one 8-layer decoder would, and the parameters
        #                    would do more as depth (--num_experts 1
        #                    --expert_num_layers 32, same params, same FLOPs).
        #       large        the mixture is doing real work.
        #     A large value with a large flow loss means they are all bad in
        #     different ways -- check the flow loss before celebrating.
        #
        # (2) DISAGREEMENT -- the old dimensionless ratio, kept for continuity.
        #     Note its upper anchor is NOT 1: for independent experts it is
        #     E[s_n]/E|N(0,1)|, i.e. 0.997 at n=2, 2/sqrt(3)=1.155 at n=4,
        #     1.193 at n=6. It drifts with num_experts, so it does not compare
        #     across configurations.
        #
        # A rising ambiguity is NOT the "mean lands between modes" failure. That
        # needs a genuinely multimodal target, and flow matching removes it
        # here: given (x_t, t, c) with deterministic demos, x_t determines the
        # noise draw, so Var(u | x_t, t, c) ~ 0 and there is one right answer.
        # The spread is estimation error, and averaging estimation error is
        # pure gain -- it is removed at the fixed rate 1 - 1/n regardless of how
        # large it is. Driving either number to zero would just make the experts
        # redundant, which is the opposite of what they are for.
        #
        # READ BOTH WITH THE STEP NUMBER. adaLN-Zero makes every residual branch
        # start at zero, so at init each expert IS the identity map and both
        # read EXACTLY 0.000 -- "not yet differentiated", not "in agreement".
        if record:
            with torch.no_grad():
                st = stacked.float()
                w = weights.float().unsqueeze(-1).unsqueeze(-1)
                # Biased (weighted) second moment, to match the identity above.
                # stacked.std() below is Bessel-corrected and does NOT.
                v_bar = (w * st).sum(dim=1, keepdim=True)
                amb_bhd = (w * (st - v_bar).pow(2)).sum(dim=1)   # (B, H, D)
                self._last_expert_ambiguity = float(amb_bhd.mean())
                # Handed to compute_loss so it can be pad-masked and bucketed by
                # horizon position with the SAME valid_cells the flow loss uses.
                # Split per position, ambiguity separates the two reasons a
                # position can be hard, which the flow loss alone cannot:
                #   amb/flow LOW   the experts agree and are all wrong together
                #                  -> the target is genuinely uncertain given the
                #                     conditioning (demonstrator timing jitter),
                #                     irreducible, and reweighting that position
                #                     spends capacity on noise.
                #   amb/flow HIGH  the experts scatter -> estimation error, which
                #                  averaging removes at 1 - 1/n and more gradient
                #                  can genuinely reduce.
                # The comment above argues Var(u | x_t, t, c) ~ 0 because x_t
                # determines the noise draw -- but that holds only if the demos
                # are deterministic given c. Where they are not, this is the
                # measurement that says so.
                self._position_ambiguity = (amb_bhd.detach()
                                            if getattr(self, "_record_position_loss", False)
                                            else None)
                spread = st.std(dim=1).mean()
                scale = st.abs().mean().clamp(min=1e-8)
                self._last_expert_disagreement = float(spread / scale)
        return v

    def router_balance_loss(self):
        """CV^2 of expert usage. Router collapse to a single expert is the known
        failure mode, and the usage vector is a batch mean, so this penalises the
        aggregate imbalance the softmax feedback loop produces."""
        u = self._last_router_usage
        if u is None or u.numel() < 2:
            return None
        return (u.var(unbiased=False) / u.mean().clamp(min=1e-8).pow(2))

    def train(self, mode: bool = True):
        super().train(mode)
        # The VLM stays in eval regardless: its base weights are frozen and its
        # norms must not update from robot batches.
        self.vision_model.eval()
        self.connector.eval()
        self.text_model.eval()
        return self

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        print(f"[wilro_moe] expert gradient checkpointing ENABLED "
              f"({len(self.experts)} experts recomputed in backward)")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
