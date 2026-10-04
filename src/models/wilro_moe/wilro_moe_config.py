"""
Config for the WILRO (VLM KV-cache → DiT cross-attention) flow matching policy.

Same encoder-decoder MoT pattern as `wiltechs_vla`, but built on the smaller
SmolVLM2-500M backbone instead of Qwen3-VL-4B (≈8× fewer VLM parameters).

  - Encoder: frozen SmolVLM2 runs ONCE per observation. K/V from the trailing
    `num_vlm_layers` text layers are cached and exposed to the DiT.
  - Decoder: `num_vlm_layers` trainable DiT layers. Each layer = causal self-attn
    over [SINK, state, prefix?, robot, latent, action] + cross-attn to one matched
    VLM KV pair + SwiGLU FFN, all modulated by adaLN-Zero from the flow-matching t.

The DiT shares the VLM's attention shape (hidden_size / num_heads / num_kv_heads
/ head_dim / intermediate_size) so cross-attention GQA aligns automatically.
"""

from dataclasses import dataclass, field
from typing import Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("wilro_moe")
@dataclass
class WilroMoEConfig(PreTrainedConfig):
    """Configuration for WILRO-MoE (SmolVLM2 KV-cache -> mixture of expert decoders)."""

    # -------- I/O structure --------
    n_obs_steps: int = 1
    horizon: int = 4
    n_action_steps: int = 4

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # -------- Image processing --------
    vision_input_size: int = 384

    # -------- VLM backbone --------
    num_cameras: int = 3
    # DiT depth = number of trailing VLM layers whose KV cache the DiT
    # cross-attends to. The VLM itself always runs ALL of its layers — this
    # field controls only how many of its KV pairs are consumed.
    # (Field name kept for backwards-compat with saved configs.)
    num_vlm_layers: int = 16

    # Which VLM layers' KV the DiT sources from:
    #   "last"    — the trailing `num_vlm_layers` layers (VLM[V-D..V-1]).
    #               All highly next-token-specialised; uses the most refined
    #               semantics but no multi-scale signal.
    #   "stride2" — evenly spaced every other layer, end-anchored so the final
    #               (most refined) layer is always included: VLM[1,3,..,V-1].
    #               Gives the DiT multi-scale features — shallow DiT layers read
    #               shallow VLM layers (local/token-level), deep DiT layers read
    #               deep VLM layers (abstract/task-level).
    #   "custom"  — use exactly the layer indices in `kv_capture_layers`. The DiT
    #               depth becomes len(kv_capture_layers) (overrides num_vlm_layers
    #               as the depth source). Indices are sorted ascending; DiT layer
    #               j reads the j-th smallest index.
    # NOTE: switching this is NOT resume-compatible — each DiT layer's cross-attn
    # is trained against a specific VLM layer's statistics.
    kv_capture_strategy: str = "last"

    # Explicit VLM layer indices for kv_capture_strategy="custom" (0-based, each
    # in [0, total_VLM_layers)). Ignored for "last"/"stride2". Example for a
    # 32-layer VLM: [3, 7, 11, 15, 19, 23, 27, 31].
    kv_capture_layers: list = field(default_factory=list)

    # Selective camera list for vision token construction.
    cameras_for_vision_state_concat: list[str] = field(default_factory=lambda: [
        'observation.images.front',
        'observation.images.gripper',
        'observation.images.right',
    ])

    # -------- State / action dims --------
    state_dim: int = 7
    action_dim: int = 7

    # -------- DiT architecture --------
    # `d_model` is forced to match the VLM hidden dim at construction time
    # (cross-attention requires both sides to share embedding dim). The DiT
    # also inherits num_heads / num_kv_heads / head_dim / intermediate_size
    # from the VLM's text config, so GQA alignment is automatic.
    # SmolVLM2-500M hidden size is 960.
    d_model: int = 960

    # Dropout used inside DiT layers (self-attn, cross-attn, FFN output).
    dropout: float = 0.1

    # -------- Flow matching sampling --------
    num_inference_steps: int = 10
    noise_temporal_correlation: float = 0.0
    # INFERENCE ONLY -- scales x_1, the ODE's starting point. 1.0 is the trained
    # setting and anything else is off-distribution for the network, so this is a
    # PROBE, not a tuning knob: shrinking x_1 pulls the sampled chunk toward the
    # conditional mean E[a|c], which is the cheap one-line stand-in for averaging
    # K independent draws. It is deliberately NOT applied in compute_loss, which
    # shares sample_noise() -- training must stay at 1.0.
    #
    # Worth it because the 2026-09-14 policy_seed pairing showed HALF of this
    # policy's goal failures are lost coin flips (21/42 rescued by nothing but a
    # different noise stream). If sampling variance costs points, this says so in
    # one eval. Because it SCALES the same draws rather than redrawing them, a
    # run at 0.9 is still paired with the 1.0 baseline and McNemar applies.
    sample_noise_scale: float = 1.0

    # -------- Temporal ensembling (inference only) --------
    # horizon=64 with n_action_steps=2 means every timestep has already been
    # predicted by up to 32 chunks, each from its own noise draw and its own
    # observation, and 31 of them are thrown away. Averaging them costs ZERO
    # extra forward passes -- the draw cadence stays n_action_steps -- and the
    # variance of the emitted action falls as 1/K_eff.
    #
    # Worth it because the 2026-09-14 policy_seed pairing put HALF of this
    # policy's goal failures down to the noise draw alone (21/42 rescued, 93%
    # of episodes win at least once), i.e. ~14 points of goal sit in sampling
    # variance. sample_noise_scale captured 2.0 of them by shrinking ONE draw;
    # this averages many.
    #
    # w_i = exp(-coeff * age_in_steps). 0.0 = OFF (bit-identical to the old
    # deque path). Large coeff -> only the newest chunk survives, which IS the
    # old path. ~0.01 is near-uniform over the 64-step window, ~0.1 gives
    # K_eff ~ 10 and keeps the ensemble fresh.
    temporal_ensemble_coeff: float = 0.0

    # -------- Stall escape (inference only) --------
    # Every intervention in this project that lowered sampling variance made
    # the DEEPEST freezes worse: sample_noise_scale 0.8 took goal's max still
    # streak 64 -> 114 chunks, and the 512 run's long failures reached 146
    # chunks = 29.2 s motionless. Noise is what escapes a stall, so averaging
    # it away is expected to inherit that cost.
    #
    # This raises the noise scale for the envs that have stopped moving, and
    # only those. The threshold is RELATIVE to the episode's own largest
    # observed state step, so it needs no units and no dataset stats.
    # 0.0 = OFF. UNTESTED -- it is a hypothesis from the motion column, not a
    # measured result.
    stall_noise_scale: float = 0.0
    stall_rel_threshold: float = 0.1   # "still" = step < this x the episode max
    stall_patience: int = 5            # consecutive still chunks before firing

    # Flow-matching TIME sampling. "uniform" (default) spends equal capacity at
    # every noise level; "lognormal" (SD3-style logit-normal) biases toward LOW t
    # — t≈0 is x_t≈actions, where the FINE action detail that sets placement
    # precision is denoised. A negative mean shifts mass toward 0.
    time_sampling: str = "uniform"          # "uniform" | "lognormal"
    time_lognormal_mean: float = -0.5       # <0 => bias toward low t (fine detail)
    time_lognormal_std: float = 1.0

    # Per-dimension and positional loss weights.
    action_dim_weights: list = field(default_factory=list)
    pos_decay_lambda: float = 0.1
    future_steps_weight: float = 0.3

    # Phase weighting: up-weight the flow-matching loss on the precision-critical
    # frames around a gripper open<->close transition (grasp / release). Uniform
    # MSE dilutes those few frames among many easy transport frames; concentrating
    # capacity there sharpens the placement the policy keeps fumbling. Weight 1.0
    # = OFF (no behavior change). Assumes the gripper is one action channel
    # (LIBERO OSC: last dim). Folded into the loss denominator so it reweights
    # rather than rescales — LR is unaffected.
    gripper_phase_weight: float = 1.0       # >1 up-weights; 1.0 disables
    gripper_action_index: int = -1          # gripper channel in the action vector
    gripper_transition_window: int = 2      # frames each side of a transition
    gripper_transition_thresh: float = 0.5  # min |Δgripper| to count as a transition

    # -------- Training presets --------
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 1500

    # -------- Robot visual cross-attention (spatial grounding) --------
    # Robot CA uses intermediate hidden states from the VLM's own SigLIP ViT
    # encoder (with LoRA adapters for robot-domain adaptation). No separate
    # ResNet model — features are extracted during the VLM forward pass, with
    # natural language-vision alignment from SigLIP's contrastive pretraining.
    #
    # When True, each DiT layer has an ADDITIONAL cross-attention sublayer
    # where action queries attend directly to high-resolution SigLIP features
    # (~729 tokens @ 384x384). This provides fine-grained spatial grounding
    # for precise object localization in spatial reasoning tasks.
    #
    # Architecture:
    #   - DiT layer: self-attn + VLM cross-attn + Robot cross-attn + FFN
    #   - adaLN-Zero: 12 modulation vectors (4 sublayers × 3) vs 9 (3 × 3)
    #   - Additional params: robot_ca_q/k/v/o_proj per DiT layer
    #   - Robot features from SigLIP ViT intermediate layer (layer_offset)
    use_vision_ca: bool = True
    # Which intermediate layer of SigLIP ViT to use for Robot CA features.
    # -1 = last layer (most semantic), -3 = third-to-last (more spatial detail).
    # SigLIP ViT has ~27 layers in SmolVLM2-500M. -3 gives the best trade-off
    # between spatial resolution and semantic richness.
    vlm_vision_layer_offset: int = -3

    # Where Robot CA's K/V actually come from.
    #
    #   "vlm" -- SigLIP ViT layer `vlm_vision_layer_offset`. Base
    #       frozen; only the LoRA adapters and the connector train (~0.39M in
    #       the robot-visual path). This is what ships and what every 2026-08/09
    #       eval measured.
    #   "resnet"           -- a separate, fully trainable ResNet-18 truncated
    #       after layer3. 3.03M at out_dim 960, MEASURED -- not the 11.7M of a
    #       stock ResNet-18, because layer4 is 72% of that and is excluded. The
    #       "~11M" in the notes and in interleaved's ARCHITECTURE.md is the
    #       stock figure and overstates this by ~4x, which matters: a rank-64
    #       vision LoRA over 27 layers is 5.31M, i.e. ALREADY LARGER than this.
    #       So if this source helps, raw trainable count is not the reason --
    #       what differs is the KIND of pathway: full-resolution pixels (256px
    #       native, stride 16 -> a 16x16 map), ImageNet init, and no frozen
    #       semantic tower underneath. This was the source until 2026-07-06 (2446dbe
    #       swapped it, 18fa4de deleted the encoder) and is still what
    #       wiltechs_moe uses; removing it there cost 34 points of spatial
    #       success (92 -> 58). wilro's own 82.5 on 2026-06-21 predates the swap
    #       and the replacement was never A/B'd.
    #
    # This REPLACES the source, it does not add a second pathway alongside it.
    # A parallel second visual encoder has been measured getting gated off by
    # the optimizer on the sibling (wiltechs_x wrist encoder: 1e-3 -> 6.2e-4,
    # confirmed twice), so "add and let the model choose" is not a neutral
    # design -- it reliably chooses the pathway that is already trained.
    vision_token_source: str = "vlm"

    # ResNet source only. `resnet_tokens` is the pooled grid per camera.
    #
    # The default here is deliberately NOT moe's 16. At input_size 224 a 14x14
    # map pooled to 4x4 makes each token cover 64 native px of a 256px LIBERO
    # frame -- HALF the granularity of the frozen VLM's 32 px merged patches,
    # for a module whose stated purpose is the precision the ViT cannot reach.
    # input_size 256 is the native frame (no resample) giving a 16x16 map:
    #   64 tok  -> 32 px/token  (parity with the VLM)  <- default
    #  144 tok  -> 21.3 px/token
    #  256 tok  -> 16 px/token  (ceiling; every feature cell kept)
    # Cost is per DiT layer -- Robot CA runs in all `num_vlm_layers` of them --
    # and scales with num_cameras, so 3 x 256 is 768 extra K/V per layer.
    resnet_tokens: int = 64
    resnet_input_size: int = 256
    # "avg" = adaptive average pooling (what moe runs). "attn" = AttentionPool2d,
    # learned queries seeded to the position grid. attn cannot honour a per-call
    # token override, which is why avg stays the default.
    resnet_pool: str = "avg"
    # Which cameras get the ResNet. Empty = all of
    # `cameras_for_vision_state_concat`.
    resnet_cameras: list[str] = field(default_factory=list)
    # Per-camera token override: cameras listed here emit `resnet_fine_tokens`
    # instead of `resnet_tokens`. 0 disables. One shared backbone serves
    # both grids at no parameter cost -- RobotVisualEncoder.forward takes an
    # out_tokens override and only the pooling depends on it.
    #
    # This exists so "give the wrist more resolution" does not have to be
    # bundled with "drop the other camera's CNN pathway": those are two changes,
    # and the sibling's 34-point ablation says the pathway as a whole is
    # load-bearing. It is also what the 2026-06/07 checkpoints used, under the
    # names gripper_camera / gripper_encoder_tokens.
    resnet_fine_cameras: list[str] = field(default_factory=list)
    resnet_fine_tokens: int = 0

    # ---- Legacy aliases: checkpoints written 2026-06-30 .. 2026-07-06 ----
    # That window is the only one where the ResNet and Robot CA both existed
    # (b3b89f1 added Robot CA on 06-30; 2446dbe/18fa4de replaced the ResNet on
    # 07-06). Their config.json carries three field names that no longer exist,
    # and draccus refuses the whole file on an unknown key -- so a checkpoint
    # that trains fine becomes unloadable, the same failure mode as lora_alpha.
    # Accepted here and translated in __post_init__; None means "not a legacy
    # config, do not translate".
    use_robot_cnn: Optional[bool] = None
    gripper_camera: Optional[str] = None
    gripper_encoder_tokens: Optional[int] = None

    # ---- Renamed 2026-09-05. Old name -> new name, one for one. ----
    # The old scheme called everything "robot": `robot_ca_source="vlm"` read as
    # "the robot cross-attention's source is the VLM", right next to a DIFFERENT
    # cross-attention that is also to the VLM (its text KV cache). Now the two
    # sources are named for what they are -- `vlm` or `resnet` -- and the
    # sublayer that consumes them is `vision_ca`, not `robot_ca`.
    #
    # EVERY dataclass field is serialised into config.json, so every wilro
    # checkpoint ever written carries the old spellings -- including runs still
    # training right now. All of them are accepted and translated.
    #
    # The state_dict is deliberately NOT renamed: robot_ca_q/o/norm,
    # robot_ca_k_proj/v_proj and robot_visual_encoder are module attributes, so
    # renaming them would invalidate every checkpoint, and six other trainers
    # read `robot_visual_encoder` by name. Weights keep the old spelling; the
    # config, CLI and docs use the new one.
    robot_ca_source: Optional[str] = None
    use_robot_ca: Optional[bool] = None
    robot_vlm_layer_offset: Optional[int] = None
    robot_encoder_tokens: Optional[int] = None
    robot_encoder_input_size: Optional[int] = None
    robot_encoder_pool: Optional[str] = None
    robot_cnn_cameras: Optional[list] = None
    robot_cnn_fine_cameras: Optional[list] = None
    robot_cnn_fine_tokens: Optional[int] = None
    robot_cnn_motion_tokens: Optional[int] = None
    robot_cnn_motion_stride: Optional[int] = None

    # -------- Temporal input (Stage B / Stage C) --------
    # wilro has no temporal input of any kind: `_encode_images` takes imgs[:, -1]
    # and `_suffix_pass` slices state_tok[:, -1:], so `--n_obs_steps` has never
    # changed anything the model sees. These two flags are what turn that off.
    #
    # B: keep every state frame instead of slicing to the last one. The leak
    #    control is already run (notes/wiltechs_x_ablations.md): the momentum
    #    shortcut sits 33x above the model's own residual, so this channel is
    #    not a shortcut, and a four-condition dose-response (frozen < noise <
    #    shuffled < real, Cochran-Armitage z=4.77, p=1.8e-06) says it carries
    #    real information. The counter-risk is the same file's task 5, where
    #    three independent corruptions of the window each cut time-to-success
    #    195 -> ~110 steps: the window can sustain a dithering loop. wilro's
    #    policy_chunks pins the 700 cap on exactly T1/T4/T5/T8/T9, which is that
    #    signature -- so read this per task, not in aggregate.
    use_state_history: bool = False

    # C: a second, older camera frame through the SAME ResNet backbone, with the
    #    FEATURE MAPS differenced and pooled to this many extra tokens. 0 off.
    #    The VLM still sees one frame -- it is 40.8% of step time and semantics
    #    do not change in 100ms; what changes is motion, which is the ResNet's
    #    job. Requires vision_token_source="resnet".
    resnet_motion_tokens: int = 0
    # How many frames back the second frame is drawn from. At 10Hz demos and
    # n_action_steps=2 the policy re-plans every 200ms, so 1 frame = 100ms is
    # the natural pairing.
    resnet_motion_stride: int = 1

    # -------- Mixture of experts --------
    # N independent decoders, each cross-attending to its OWN contiguous band of
    # VLM layers, mixed by a router. Ported from wiltechs_moe, which is this
    # repo's best LIBERO result (92 spatial).
    #
    # num_experts * expert_num_layers must not exceed the VLM's layer count,
    # because the bands are disjoint. SmolVLM2-500M has 32 text layers where
    # Qwen3-VL-4B has 36, so wiltechs_moe's 4 x 9 = 36 does NOT fit here and
    # 4 x 8 = 32 does -- exactly, with every layer used.
    num_experts: int = 4
    expert_num_layers: int = 8
    # 0 = the VLM's hidden size. At 960 (== hidden) the expert self-attention
    # inherits the VLM's 15/5/64 geometry directly and needs no GQA re-derivation.
    dit_hidden_size: int = 960
    # Explicit VLM layer indices to capture. Empty = all of them, which is what
    # 4 x 8 = 32 wants. Must be divisible by num_experts.
    vlm_capture_layers: list = field(default_factory=list)

    # Per-expert read of the SHARED vision tokens: a bottleneck MLP applied to
    # the vision span of each expert's own copy of the sequence. 0 disables.
    #
    # The tokens are one set read by every expert, so the encoder that produced
    # them receives a router-weighted sum of four demands and has to compromise.
    # This lets the trunk stay generic while each expert learns its own read,
    # for ~0.5M per expert at dim 256 against the ~3M (and 4x the convolution)
    # that four separate ResNets would cost.
    #
    # Two things to know before turning it on:
    #
    #  * It is PARTLY redundant with what the experts already have. Each
    #    expert's self-attention value projection is already a per-expert linear
    #    read of these tokens. The adapter adds a nonlinearity and changes the
    #    tokens' value in the RESIDUAL STREAM, which every later layer sees, so
    #    it is not pure duplication -- but the marginal capacity is smaller than
    #    the parameter count suggests.
    #
    #  * It makes router collapse MORE expensive, not less. With shared tokens a
    #    collapsed router still trains the trunk through whichever expert is
    #    live; with per-expert adapters the unused ones receive no gradient at
    #    all and sit at init, so an expert coming back later reads vision
    #    through an untrained map. That is why the adapter is a ZERO-INIT
    #    RESIDUAL: untrained means identity, not garbage, and the gate doubles
    #    as the instrument that catches suppression.
    #
    # Default off on purpose: wiltechs_moe scores 92 WITHOUT it, so switching it
    # on for the first wilro_moe run would make that number unattributable.
    resnet_expert_adapter_dim: int = 0

    router_temperature: float = 1.0
    # 0 = soft mixture over all experts. >0 keeps only the top-k.
    router_top_k: int = 0
    # Penalise uneven expert usage. Router collapse to one expert is the known
    # failure mode; MoERouter also injects fixed N(0, 0.5) logit noise in
    # training for the same reason.
    router_balance_weight: float = 0.1


    # -------- Latent tokens: INERT in this model --------
    # wilro's task-conditional latents (an MLP over pooled language). The MoE
    # decoder does not build them -- _generate_latents returns None -- so this
    # is here only because the loss code is shared with wilro, which calls it.
    # Not to be confused with wiltechs_moe's "thought" tokens, a QFormer over
    # real VLM KV, which this model does not have either (removed 2026-09-05).
    num_latent_tokens: int = 0

    # -------- Vision token dropout (regularizer) --------
    vision_dropout_prob: float = 0.15

    # -------- Auxiliary contrastive loss (language forcing) --------
    contrastive_loss_weight: float = 0.1
    contrastive_margin: float = 0.05
    contrastive_hard_negatives: bool = False

    # -------- Instruction surface-form augmentation --------
    # Draw a different phrasing of the same instruction per sample per step, so
    # the surface string stops being a usable key. Measured on the sibling
    # (wiltechs-x-114k, libero_spatial T7): 60% on its own instruction, 0% on a
    # PARAPHRASE of that same instruction -- it had memorised the ~40 strings
    # and was retrieving, not reading. The table lives in src/libero_paraphrase.py.
    #
    # The draw happens in `_encode_language` only, so the contrastive hinge
    # still sees the CANONICAL strings from the batch and keeps deciding "same
    # instruction or not" by exact equality. Paraphrasing before that check
    # would make two phrasings of one task read as two tasks, and the hinge
    # would penalise the model for agreeing with itself in other words --
    # fighting precisely what this is for.
    paraphrase_augment: bool = False
    # Cap on variants per instruction (0 = all). The table has 5-7 each.
    paraphrase_limit: int = 8
    # JSON table overriding the built-in one, for instructions it does not
    # cover. `python -m libero_paraphrase --dataset_id ... --out f.json`, then
    # hand-edit. Templates are NOT consulted at training time.
    paraphrase_file: str = ""
    # The trainer preflight refuses to start when any instruction has fewer
    # variants than this. Partial augmentation is worse than none: the unvaried
    # tasks keep surface form as a key, and the run answers nothing.
    paraphrase_min_variants: int = 5

    # -------- Action prefix for async execution (paper Sec 2.2.2) --------
    # Max number of clean action prefix steps to condition on. During training,
    # Δt_c is sampled from {0, 1, ..., max_action_prefix_steps}. When > 0,
    # earlier actions are prepended to the noisy action sequence in DiT.
    # 0 disables (synchronous execution mode).
    max_action_prefix_steps: int = 0

    # Λ-shape attention mask: noisy action tokens of later timesteps cannot
    # attend to the conditioned action prefix, forcing them to rely on visual
    # and language signals. (paper Fig 4)
    lambda_mask_window: int = 3

    # -------- LoRA (SigLIP ViT vision + text_model) --------
    # LoRA adapters on the last N layers of SigLIP ViT enable the vision
    # encoder to adapt to robot-domain features (gripper aperture, object
    # distance, contact state) while preserving SigLIP's contrastive
    # language-vision alignment.
    # LoRA adapters on text_model layers enable language adaptation for
    # robot-specific instructions and spatial grounding.
    # Base weights stay frozen; only LoRA params are trainable.
    lora_rank: int = 16
    # float, not int: LoRALinear uses it only as the ratio alpha/rank, and
    # the trainer's default of 2 x rank is written out as e.g. 128.0. Declared
    # as int, draccus refused to load any checkpoint carrying such a value --
    #   DecodingError: `lora_alpha`: Couldn't parse '128.0' into an int
    # so a run trained fine and then could not be evaluated. Widening also
    # reads the older checkpoints, whose value is a plain 32.
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    vision_lora_num_layers: int = 8  # Last 8 layers of SigLIP ViT get LoRA
    text_lora_num_layers: int = 0    # DISABLED — encoder-decoder requires detached VLM outputs

    # -------- Resume bookkeeping --------
    training_step: int = 0
    training_epoch: int = 0
    current_lr: float = 0.0
    training_steps_total: int = 0

    def __post_init__(self):
        """Accept every spelling a wilro config.json has ever used.

        Two generations of aliases: the 2026-06/07 fields removed when the
        ResNet was, and the 2026-09-05 rename. Both translate onto the current
        names. Explicit current values win, so a shim can never override an
        intentional setting -- only fill in what the caller left at its default.

        The aliases are MIRRORED back at the end, never left as None. Setting
        them to None and saving is what broke: draccus writes `"use_robot_ca":
        null` into config.json, and on a draccus older than 0.10 the field
        resolves to decode_bool rather than the union decoder, so the file this
        class wrote is a file this class cannot read --

            DecodingError: `use_robot_ca`: Couldn't parse 'None' into a bool

        Mirroring keeps every alias holding the same value as the field it
        aliases, which makes a re-read a no-op and means no null is ever
        serialised. Round-tripped in tests, which the None version was not.
        """
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()

        RENAMED = (
            ("robot_ca_source", "vision_token_source"),
            ("use_robot_ca", "use_vision_ca"),
            ("robot_vlm_layer_offset", "vlm_vision_layer_offset"),
            ("robot_encoder_tokens", "resnet_tokens"),
            ("robot_encoder_input_size", "resnet_input_size"),
            ("robot_encoder_pool", "resnet_pool"),
            ("robot_cnn_cameras", "resnet_cameras"),
            ("robot_cnn_fine_cameras", "resnet_fine_cameras"),
            ("robot_cnn_fine_tokens", "resnet_fine_tokens"),
            ("robot_cnn_motion_tokens", "resnet_motion_tokens"),
            ("robot_cnn_motion_stride", "resnet_motion_stride"),
        )
        DEFAULTS = {
            "vision_token_source": "vlm", "use_vision_ca": True,
            "vlm_vision_layer_offset": -3, "resnet_tokens": 64,
            "resnet_input_size": 256, "resnet_pool": "avg",
            "resnet_cameras": [], "resnet_fine_cameras": [],
            "resnet_fine_tokens": 0, "resnet_motion_tokens": 0,
            "resnet_motion_stride": 1,
        }
        # Which aliases actually arrived. A mirrored re-read has them ALL set,
        # which is how the use_robot_cnn=False guard below tells a genuine
        # 2026-06/07 config from a config this class wrote.
        present = {o for o, _ in RENAMED if getattr(self, o) is not None}
        moved = []
        for old_name, new_name in RENAMED:
            v = getattr(self, old_name)
            if v is None:
                continue
            if new_name == "vision_token_source" and v == "vlm_intermediate":
                v = "vlm"
            if getattr(self, new_name) == DEFAULTS[new_name] and v != DEFAULTS[new_name]:
                setattr(self, new_name, v)
                moved.append(f"{old_name}={v!r} -> {new_name}")
        if self.vision_token_source == "vlm_intermediate":
            self.vision_token_source = "vlm"
            moved.append("vision_token_source 'vlm_intermediate' -> 'vlm'")

        # 2026-06/07 fields, removed when the ResNet was.
        if self.use_robot_cnn is not None:
            if not self.use_robot_cnn and "robot_ca_source" not in present:
                raise ValueError(
                    "use_robot_cnn=False came from a checkpoint whose DiT has no "
                    "vision tokens at all. The current model has no way to express "
                    "that -- vision_token_source selects WHICH source feeds the "
                    "tokens, not whether they exist. This checkpoint needs the "
                    "code of its own era.")
            if self.use_robot_cnn and self.vision_token_source != "resnet":
                self.vision_token_source = "resnet"
                moved.append("use_robot_cnn=True -> vision_token_source='resnet'")
        if self.gripper_camera and self.gripper_encoder_tokens \
                and not self.resnet_fine_cameras:
            self.resnet_fine_cameras = [self.gripper_camera]
            self.resnet_fine_tokens = int(self.gripper_encoder_tokens)
            moved.append(
                f"gripper_camera/gripper_encoder_tokens -> "
                f"resnet_fine_cameras={self.resnet_fine_cameras} "
                f"/ resnet_fine_tokens={self.resnet_fine_tokens}")

        if moved:
            print("[wilro] legacy config translated:")
            for m in moved:
                print(f"          {m}")

        # MIRROR, never None: see the docstring. Every alias ends up holding the
        # value of the field it aliases, so nothing serialises as null and a
        # re-read translates each name onto itself.
        for old_name, new_name in RENAMED:
            setattr(self, old_name, getattr(self, new_name))
        self.use_robot_cnn = (self.vision_token_source == "resnet")
        self.gripper_camera = (self.resnet_fine_cameras[0]
                               if self.resnet_fine_cameras else "")
        self.gripper_encoder_tokens = int(self.resnet_fine_tokens)

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("Provide at least one image feature or env state.")
        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, ft in self.image_features.items():
                if ft.shape != first_ft.shape:
                    raise ValueError(
                        f"`{key}` shape {ft.shape} does not match `{first_key}` {first_ft.shape}"
                    )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=90000,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.01,
        )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None