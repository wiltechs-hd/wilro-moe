"""LIBERO evaluation for WILRO-MoE checkpoints.

    python src/eval_libero.py \
        --checkpoint ISdept/wilro-wilromoe-8x4-22k-obs2 \
        --suites libero_spatial libero_object libero_goal libero_10 \
        --episodes 20 --n_action_steps 2

READ THE `min` COLUMN, NOT THE AVERAGE. ARCHITECTURE.md §1: stage-B RL recovers
a task sitting at 10% and can do nothing with one sitting at 0, because a binary
reward has no gradient where every rollout fails. 93% average with a per-task
floor of 15% is a better stage-A checkpoint than 95% with two zeros. The gate
this prints is `avg >= 93 AND min > 5`.

Three things this harness pins that a naive eval gets wrong, each of which has
already cost this repo a set of non-comparable numbers:

  * `control_freq=10`. The LIBERO demos are 10 Hz and stock robosuite is 20, so
    a delta-EEF action sized for 1/10 s is held for 1/20 s instead and moves
    half as far. Numbers taken at 20 Hz do not transfer (rollout 0.86 against
    eval 77.5% in this repo's own RL runs).
  * The canonical 50 initial states. lerobot's LiberoEnv.reset() writes the init
    state and THEN lets robosuite re-sample the placement initializer over it,
    serving layouts 3-10x more spread out than the ones the demos were recorded
    on. `libero_env_fixed.patch_lerobot_libero()` restores LIBERO's own order.
    `--stock_init` runs without the fix, for an A/B on one checkpoint.
  * The proprioceptive HISTORY. Training feeds `observation.state` as
    (B, motion_history_len, D) via delta_timestamps, and MotionVectorEncoder
    takes first differences over it. An eval that passes the single current
    frame leaves the model with an all-zero motion signal -- it will still run,
    score lower, and say nothing about why. See `StateHistory` below.

Few-step inference is a claim, not a given: `--num_inference_steps` defaults to
the config's 4. Re-running at 16 measures whether the shortcut consistency term
actually made 4 NFE valid. A large gap means it did not, and the flow objective
-- not the policy -- is what to fix.
"""
from __future__ import annotations

import os

# Must precede any robosuite/mujoco import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Same reason, one layer up: LIBERO's env_wrapper imports matplotlib.cm, and
# matplotlib resolves MPLBACKEND at import time. A notebook exports
# MPLBACKEND=module://matplotlib_inline.backend_inline, which is only importable
# inside the notebook's OWN interpreter -- run this script from a Colab cell
# against any other environment and matplotlib raises before LIBERO loads. This
# is a headless eval that draws nothing, so agg is the right backend; a
# deliberate non-inline choice is left alone.
_mpl = os.environ.get("MPLBACKEND", "")
if not _mpl or "inline" in _mpl:
    os.environ["MPLBACKEND"] = "agg"

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from env_fingerprint import fingerprint, fingerprint_line

sys.path.insert(0, str(Path(__file__).resolve().parent))

from libero_env_fixed import patch_lerobot_libero

# One harness, several policies -- deliberately NOT a second script.
#
# Everything that makes two numbers comparable lives in this file: the
# canonical-50 init states (patch_lerobot_libero), control_freq=10, per-task
# policy seeding, the paired per-episode vectors. A copy for another model
# would be a second place for those to drift, and this repo has already paid
# for exactly that -- the sibling's 92% was produced by a script that is no
# longer in the tree, so it cannot be re-run against the current fixes at all.
#
# Adding a model here is two lines and it inherits every fix.
POLICIES = {
    "wilro_moe":    ("models.wilro_moe.wilro_moe_policy", "WilroMoEPolicy"),
    # Off-the-shelf policies, so a CANDIDATE TEACHER can be scored on the same
    # harness before anyone distils from it. Published LIBERO numbers are not
    # comparable to this repo's: everything measured before 2026-08-03 used
    # lerobot's reset ordering, which serves layouts far wider than the
    # canonical 50 (see the module docstring). A teacher that does not clearly
    # beat the student HERE is not a teacher.
    "pi0":          ("lerobot.policies.pi0.modeling_pi0", "PI0Policy"),
    "pi05":         ("lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
    "groot":        ("lerobot.policies.groot.modeling_groot", "GrootPolicy"),
}


def _register_configs():
    """Importing a config module is what registers its `type` string, and
    PreTrainedConfig.from_pretrained resolves the checkpoint by that string.
    Failures are per-model and non-fatal, WiltechsX included: this harness is
    routinely run from a git worktree pinned to an older commit, to score a
    checkpoint against the code of its own era, and in such a tree the other
    models simply do not exist yet. A top-level import of any one of them makes
    the harness unusable exactly where it is most needed."""
    for mod in ("models.wilro_moe.wilro_moe_config",
                "lerobot.policies.pi0.configuration_pi0",
                "lerobot.policies.pi05.configuration_pi05",
                "lerobot.policies.groot.configuration_groot"):
        try:
            __import__(mod)
        except Exception:
            pass


_register_configs()


def _policy_class(kind: str):
    import importlib
    if kind not in POLICIES:
        raise SystemExit(
            f"checkpoint declares policy type {kind!r}, which this harness does "
            f"not know.\n    Known: {', '.join(sorted(POLICIES))}\n"
            f"    Add it to POLICIES in {Path(__file__).name} -- two lines, and "
            f"it inherits every eval fix.")
    mod, cls = POLICIES[kind]
    return getattr(importlib.import_module(mod), cls)


def _policy_cameras(cfg) -> list[str]:
    """Camera keys the checkpoint expects, from the checkpoint itself.

    Each model names this differently, and the name matters because the
    ENCODER iterates its own field and SKIPS keys the batch does not carry
    (wilro_model `_encode_images`: `if cam_key not in batch: continue`). Read
    input_features instead and the guard below checks a list the encoder never
    consults -- a checkpoint whose camera field still held another robot's keys
    would pass the guard and then encode ZERO cameras, scoring a blind policy.
    So ask each model for ITS field first, in the order they were added, and
    fall back to input_features only when a config declares none of them.
    """
    for attr in ("cameras_for_vlm",                    # wiltechs_x / moe / vla
                 "cameras_for_vision_state_concat"):   # wilro
        cams = list(getattr(cfg, attr, None) or [])
        if cams:
            return cams
    feats = getattr(cfg, "input_features", None) or {}
    cams = [k for k, v in feats.items()
            if getattr(getattr(v, "type", None), "name", "") == "VISUAL"]
    if cams:
        return sorted(cams)
    raise SystemExit(
        "cannot tell which cameras this checkpoint expects: its config has "
        "neither cameras_for_vlm nor VISUAL input_features.")


def _git_commit() -> str | None:
    """Which build produced this JSON. Cheap, and the alternative is guessing
    from which keys happen to be present in the file."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                            "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Proprioceptive history
# ---------------------------------------------------------------------------
class StateHistory:
    """Rolling (T, D) window of `observation.state`, one per env.

    Training builds this with delta_timestamps
    (`[-i*ft for i in range(n_obs_steps)]`), so the model has always seen T
    frames. At reset there is only one, and LeRobot's own left-padding
    convention repeats the earliest frame -- MotionVectorEncoder does exactly
    that internally, so seeding the deque full of the reset state reproduces it
    without relying on the encoder's fallback.
    """

    MODES = ("real", "frozen", "shuffled", "noise")

    def __init__(self, n_envs: int, history_len: int, mode: str = "real",
                 seed: int = 0):
        self.t = max(1, int(history_len))
        self.buf = [deque(maxlen=self.t) for _ in range(n_envs)]
        if mode not in self.MODES:
            raise SystemExit(f"--history_mode must be one of {self.MODES}")
        # `shuffled` permutes the OLDER T-1 frames, leaving frame -1 alone
        # because that slot is the current proprioceptive reading. At T=2 that
        # leaves exactly ONE frame to permute, and a permutation of one element
        # is the identity -- the run returns results BIT-IDENTICAL to `real` and
        # reads as "the state window does not matter" when in fact nothing was
        # ablated. Refuse rather than return that.
        if mode == "shuffled" and self.t < 3:
            raise SystemExit(
                f"--history_mode shuffled is a NO-OP at n_obs_steps={self.t}: it "
                f"permutes the older T-1 = {self.t - 1} frame(s), and permuting "
                f"{self.t - 1} element(s) is the identity. It would return "
                f"bit-identical results to --history_mode real.\n"
                f"  At T=2 use --history_mode noise (replaces the older frame "
                f"with the newest plus Gaussian jitter at the window's own "
                f"per-dim std: motion MAGNITUDE survives, DIRECTION dies, and "
                f"unlike `frozen` it asserts nothing).\n"
                f"  --history_mode frozen also works at T=2 (velocity "
                f"identically zero) but is in-distribution -- every episode's "
                f"first inference call already sees it -- so a NULL result "
                f"under frozen is ambiguous. Read `noise`; use `frozen` as the "
                f"second reading.")
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def reset(self, i: int, state: np.ndarray):
        self.buf[i].clear()
        for _ in range(self.t):
            self.buf[i].append(np.asarray(state, dtype=np.float32))

    def push(self, i: int, state: np.ndarray):
        self.buf[i].append(np.asarray(state, dtype=np.float32))

    def stack(self) -> np.ndarray:
        """-> (n_envs, T, D), after whatever ablation `mode` asks for.

        The one place the window is assembled, so the one place to intervene.
        See ARCHITECTURE.md 8.2: the model can form `s_t - s_{t-1}` from this
        window, and under a position controller that difference IS the
        previously executed action. Demos are smooth, so extrapolating it
        explains the near horizon without reading the image -- and at
        `n_action_steps=2` the near horizon is the ONLY part ever executed.

          real      untouched.
          frozen    newest frame repeated. Velocity is identically zero. This
                    is NOT out of distribution: `reset` builds exactly this,
                    so every episode's first inference call already sees it.
          shuffled  the OLDER T-1 frames permuted; ordering dies, every
                    marginal survives.
          noise     the older T-1 frames replaced by the newest plus Gaussian
                    noise at the real window's own per-dim std. Motion
                    MAGNITUDE is preserved, direction is gone, and -- unlike
                    `frozen` -- the window makes no coherent claim.

        EVERY mode leaves frame -1 untouched, because the state token is
        `st[:, -1]` (wiltechs_x_model, _suffix_pass). A permutation over all T
        moves an older frame into that slot and displaces the CURRENT
        proprioceptive reading by up to T-1 frames, which is a second
        intervention on top of the intended one. Results taken before this was
        fixed under-state nothing -- they destroyed ordering AND the state
        token -- but they cannot be read against `frozen`, which never had the
        defect.

        Why three: `frozen` alone is ambiguous. It does not merely remove the
        signal, it asserts a self-consistent falsehood ("this arm has been
        still for T frames") that a phase detector can lock onto, so a
        collapse under it can mean either "the signal was needed" or "the lie
        selected the wrong mode". `noise` carries variance without either a
        true velocity or that assertion, and separates the two.
        """
        out = np.stack([np.stack(list(b)) for b in self.buf])
        if self.mode == "frozen":
            out[:] = out[:, -1:, :]
        elif self.mode == "shuffled" and self.t > 1:
            for i in range(len(out)):
                out[i, :-1] = out[i][self.rng.permutation(self.t - 1)]
        elif self.mode == "noise" and self.t > 1:
            sd = out.std(axis=1, keepdims=True)
            jitter = self.rng.normal(0.0, 1.0, out.shape) * sd
            out[:, :-1] = (out[:, -1:, :] + jitter[:, :-1]).astype(out.dtype)
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_policy(ckpt: Path, device: str, num_inference_steps: int | None,
                n_action_steps: int | None = None,
                fixed_episode_noise: bool = False,
                sample_noise_scale: float | None = None,
                router_top_k: int | None = None,
                log_routing: bool = False,
                vision_input_size: int | None = None,
                temporal_ensemble_coeff: float | None = None,
                stall_noise_scale: float | None = None,
                stall_rel_threshold: float | None = None,
                stall_patience: int | None = None):
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = str(device)
    # Guarded: these are flow-policy knobs and not every sibling has them.
    # Setting an attribute a config does not declare would silently do nothing
    # on a dataclass with slots, or silently invent a field on one without.
    def _set(name, value, flag):
        if not hasattr(cfg, name):
            raise SystemExit(
                f"{flag} was passed, but {type(cfg).__name__} has no "
                f"'{name}'. That knob does not exist for this policy.")
        setattr(cfg, name, value)

    if num_inference_steps:
        _set("num_inference_steps", int(num_inference_steps),
             "--num_inference_steps")
    if n_action_steps:
        n = int(n_action_steps)
        if n > int(cfg.horizon):
            raise SystemExit(
                f"--n_action_steps {n} exceeds the trained horizon "
                f"{cfg.horizon}: the chunk has no steps past that to execute.")
        cfg.n_action_steps = n
    if vision_input_size:
        v = int(vision_input_size)
        # The VLM tower is FROZEN, so this is a no-retrain probe: the encoder
        # runs at a different resolution and the token count changes with it
        # (SmolVLM2-500M is patch 16, pixel-shuffle 4, so tokens = (v/16/4)^2
        # per camera -- 36 at 384, 64 at its native 512). The trainable proj is
        # per-token and the experts cross-attend over a variable-length KV, so
        # it runs. Read the result ASYMMETRICALLY: an improvement is real
        # evidence, a drop is not, because everything downstream was fitted to
        # the old token count.
        if v % 64:
            raise SystemExit(
                f"--vision_input_size {v} must be divisible by 64 "
                f"(patch 16 x pixel-shuffle 4); {v // 16} patches per side is "
                f"not divisible by the shuffle factor.")
        _set("vision_input_size", v, "--vision_input_size")
    for _val, _name, _flag in (
            (temporal_ensemble_coeff, "temporal_ensemble_coeff",
             "--temporal_ensemble_coeff"),
            (stall_noise_scale, "stall_noise_scale", "--stall_noise_scale"),
            (stall_rel_threshold, "stall_rel_threshold", "--stall_rel_threshold"),
            (stall_patience, "stall_patience", "--stall_patience")):
        if _val is not None:
            _set(_name, type(getattr(cfg, _name, _val))(_val), _flag)
    if fixed_episode_noise:
        _set("fixed_episode_noise", True, "--fixed_episode_noise")
    if sample_noise_scale is not None:
        # Temperature on x_1. NOT the same experiment as
        # --fixed_episode_noise: that commits to one RANDOM draw, this moves
        # every draw toward the centre of the policy's distribution. Fixing
        # the noise cost 25 points here, which says the per-chunk lottery is
        # rescuing episodes -- but a lottery only helps when the distribution
        # is too broad, and shrinking it is the other way to answer that.
        _set("sample_noise_scale", float(sample_noise_scale),
             "--sample_noise_scale")
    kind = getattr(cfg, "type", None) or getattr(cfg, "name", "")
    print(f"policy type: {kind}")
    if router_top_k is not None:
        # TRAIN/TEST MISMATCH BY CONSTRUCTION. The model was trained with the
        # config's own top_k and with N(0, 0.5) exploration noise on the router
        # logits; inference has no noise, so it already sees a distribution it
        # never trained on. Changing k widens that gap deliberately. Nothing
        # breaks -- the mask is applied then renormalised -- but the mixture is
        # not the one the loss was minimised over.
        _k = int(router_top_k)
        _n = int(getattr(cfg, "num_experts", 0) or 0)
        if _n and not (0 <= _k <= _n):
            raise SystemExit(f"--router_top_k {_k} outside [0, num_experts={_n}]")
        print(f"[eval] router_top_k {getattr(cfg, 'router_top_k', '?')} -> {_k}"
              + (" (0 = dense, all experts weighted)" if _k == 0 else ""))
        _set("router_top_k", _k, "--router_top_k")

    policy = _policy_class(kind).from_pretrained(ckpt, config=cfg)
    if log_routing:
        m = getattr(policy, "model", None)
        if m is None or not hasattr(m, "_record_routing"):
            raise SystemExit("--log_routing needs a policy whose model carries "
                             "a routing trace; only wilro_moe has one.")
        m._record_routing = True
    policy.to(device)
    policy.eval()
    for m in policy.model.modules():                      # deterministic rollout
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return policy


def report_new_config_fields(cfg, ckpt: Path):
    """Fields the CODE declares that the checkpoint's config.json never set.

    The weight audit's exact analogue, and the more dangerous half: a config
    field added after a checkpoint was written gets its dataclass DEFAULT with
    no warning anywhere, and if that default is non-empty the model computes
    something different from what was trained.

    Real case: WiltechsMoE's `instruction_template` landed 2026-08-11 with
    `= STARVLA_COT_TEMPLATE`, a week after the checkpoint that scored 92%. That
    checkpoint's config.json has no such key, so loading it silently switched
    the model's text input to a bounding-box CoT prompt it was never trained
    on. No missing key, no shape mismatch, no warning -- just a different
    model wearing the same number.

    Values are printed so a non-empty default is visible as such; deciding
    which ones matter needs the git log, which is why this reports rather than
    refuses.
    """
    f = ckpt / "config.json"
    if not f.exists():
        return
    try:
        saved = set(json.loads(f.read_text()))
    except Exception:
        return
    import dataclasses
    if not dataclasses.is_dataclass(cfg):
        return
    new = [fl.name for fl in dataclasses.fields(cfg)
           if fl.name not in saved and not fl.name.startswith("_")]
    if not new:
        return
    print(f"[config] {len(new)} field(s) declared by the code but absent from "
          f"this checkpoint's config.json -- each took its DEFAULT:")
    for n in sorted(new):
        v = getattr(cfg, n, None)
        empty = v in (None, "", 0, 0.0, False) or (isinstance(v, (list, dict, tuple)) and not v)
        r = repr(v)
        print(f"    {n:<34s} = {r[:80] + ('...' if len(r) > 80 else '')}"
              + ("" if empty else "   <- NON-EMPTY, changes behaviour"))
    print("  Anything marked NON-EMPTY postdates this checkpoint and is active "
          "now but was not during training.\n"
          "  Override it, or evaluate against the code of that era -- see "
          "`git log -- <model dir>`.")


def report_missing_weights(policy, ckpt: Path, allow: bool):
    """Account for the tensors the checkpoint did not supply -- by requires_grad.

    lerobot logs these as one `WARNING:root:Missing key(s)` line and carries on,
    which can mean an eval silently scores a DIFFERENT model than the one that
    was trained. But "missing" alone is not the signal, and the first version of
    this function got that wrong: it flagged all 714 frozen Qwen3-VL tensors of
    a WiltechsMoE checkpoint and would have refused to run.

    Those are missing BY DESIGN. train_wiltechs_moe strips `model.vlm_model.*`
    at save time, and says why: "the encoder is always loaded by
    from_pretrained(model_id) before this point, so the checkpoint's copy is
    redundant either way". Their values in state_dict() are the correct
    pretrained weights, not an initialisation.

    So the discriminator is requires_grad, not presence:

      frozen and missing     expected -- the value came from the pretrained
                             source at construction. Summarised, not listed.
      TRAINABLE and missing  a learned weight sitting at its init. THAT is the
                             alarm, and whether it matters depends on what the
                             init is: zero contributes nothing, anything else
                             changes the model.

    The case that prompted the whole check is the benign end of that:
    `model.robot_pos_gate` is trainable, arrived in 4c06db6 (2026-08-08) after
    the checkpoint that scored 92% (2026-08-04), and is
    nn.Parameter(torch.zeros(1)) multiplying an additive term -- so at 0 it is
    exactly the model that was trained. A norm weight initialised to 1.0 would
    have looked identical in the log and would not have been.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return
    files = sorted(ckpt.glob("*.safetensors"))
    if not files:
        return
    have = set()
    for f in files:
        with safe_open(f, framework="pt") as fh:
            have |= set(fh.keys())
    sd = policy.state_dict()
    if len(have & set(sd)) < 0.2 * len(sd):
        print(f"[weights] key naming does not line up with the checkpoint "
              f"({len(have & set(sd))}/{len(sd)} matched); skipping the audit.")
        return
    grads = {k: p.requires_grad for k, p in policy.named_parameters()}
    missing = sorted(k for k in sd if k not in have)
    if not missing:
        return
    # Buffers are not parameters; grads.get(...) is False for them, which puts
    # them on the expected side. That is right -- they are constants or caches.
    frozen = [k for k in missing if not grads.get(k, False)]
    live = [(k, float(sd[k].detach().abs().max()))
            for k in missing if grads.get(k, False)]

    if frozen:
        n = sum(sd[k].numel() for k in frozen)
        print(f"[weights] {len(frozen)} FROZEN tensor(s) / {n/1e6:.0f}M params not "
              f"in the checkpoint -- expected: a frozen backbone is loaded at "
              f"construction, so the checkpoint does not carry a second copy.")
    if not live:
        print("[weights] every TRAINABLE tensor was supplied. Good.")
        return
    print(f"[weights] {len(live)} TRAINABLE tensor(s) not in the checkpoint "
          f"-- each keeps its INITIALISATION:")
    for k, m in live:
        print(f"    {k:<50s} {sd[k].numel():>10,} el   |max| {m:.3e}"
              + ("   inert (exactly 0)" if m == 0.0 else "   *** NONZERO ***"))
    if all(m == 0.0 for _, m in live):
        print("  All zero, so they contribute nothing: this is numerically the "
              "model the checkpoint was trained as.")
        return
    msg = ("Some are NONZERO, so this is not the model that was trained and the "
           "number below would not be comparable to anything.")
    if not allow:
        raise SystemExit(f"  {msg}\n  Pass --allow_missing_weights to score it "
                         f"anyway, knowing that.")
    print(f"  {msg}  --allow_missing_weights was passed; continuing.")


def load_processors(ckpt: Path, device: str, dataset_id: str | None):
    """Prefer the pipelines saved next to the weights.

    They carry the dataset statistics the policy was TRAINED against, which is
    the only correct choice: rebuilding from a dataset that has since gained
    episodes would unnormalize with different numbers than the model learned.
    `--dataset_id` exists only for checkpoints written before the trainer saved
    them, and says so loudly.
    """
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    pre_json = ckpt / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    if pre_json.exists():
        pre = PolicyProcessorPipeline.from_pretrained(
            ckpt, config_filename=pre_json.name,
            overrides={"device_processor": {"device": str(device)}})
        post = PolicyProcessorPipeline.from_pretrained(
            ckpt, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
            overrides={"device_processor": {"device": "cpu"}})
        print(f"[eval] processors loaded from {ckpt.name} (training statistics)")
        return pre, post

    if not dataset_id:
        raise SystemExit(
            f"{pre_json} is missing and no --dataset_id was given. Without the "
            f"normalization statistics the policy's inputs and outputs are on "
            f"the wrong scale and every number this script prints is noise.")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from models.wilro_moe.processor_wilro_moe import make_pre_post_processors

    print(f"[eval] WARNING: no saved processors in {ckpt}; rebuilding from "
          f"{dataset_id}. Valid only if that dataset is byte-identical to the "
          f"one trained on.")
    from models.wilro_moe.wilro_moe_config import WilroMoEConfig
    cfg = WilroMoEConfig.from_pretrained(ckpt)
    cfg.device = str(device)
    stats = LeRobotDatasetMetadata(dataset_id, revision="main").stats
    return make_pre_post_processors(cfg, dataset_stats=stats)


def patch_control_freq(control_freq: int, render_gpu: int):
    """Build LiberoEnv's OffScreenRenderEnv at `control_freq` Hz.

    Same patch as `train_wilro_rl._patch_libero_control_freq`, inlined rather
    than imported: that module is a 1200-line trainer that sets up multiprocess
    EGL at import, and an eval should not drag that in for six lines.
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from lerobot.envs.libero import LiberoEnv

    def _make_envs_task(self, task_suite, task_id: int = 0):
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        # robosuite reads render_gpu_device_id, NOT MUJOCO_EGL_DEVICE_ID.
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            control_freq=control_freq,
            render_gpu_device_id=max(render_gpu, 0),
        )
        env.reset()
        return env

    LiberoEnv._make_envs_task = _make_envs_task
    print(f"[eval] env control_freq={control_freq} Hz "
          f"({'matches the 10 Hz dataset' if control_freq == 10 else 'NOT 10 Hz — numbers will not transfer'})")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
# The inference knobs a LIBERO result is only valid under. Anything that
# outlives a single run -- a golden ticket, say -- is stamped with these and
# checked against them on reuse. max_episode_steps is deliberately NOT here:
# it is a search-budget choice, not a property of the policy.
MUST_MATCH = ("n_action_steps", "num_inference_steps", "horizon",
              "control_freq", "stock_init")


def setup_libero_env(control_freq: int = 10, render_gpu: int = 0,
                     stock_init: bool = False) -> dict:
    """Every env-side patch LIBERO needs, in one call, in this order.

    It is one function because two scripts have now shipped a bug by copying
    one of these and not the other. Without patch_lerobot_libero, reset()
    sets the init state and robosuite then re-samples the BDDL placement
    initializer over it, so rollouts land on placements ~10x wider than the
    canonical 50 -- never seen in training, reported by nobody. Without
    patch_control_freq the env runs at 20 Hz against 10 Hz data.

    Neither failure announces itself; both just make the policy look broken.
    """
    canonical = patch_lerobot_libero(enable=not stock_init)
    patch_control_freq(control_freq, render_gpu)
    print(f"[env] control_freq={control_freq}  render_gpu={render_gpu}  "
          f"init states={'canonical 50' if canonical else 'SAMPLER ORDER (stock lerobot)'}",
          flush=True)
    return {"control_freq": int(control_freq), "stock_init": bool(stock_init),
            "canonical_init_states": bool(canonical)}


def inference_config(policy, control_freq: int, max_episode_steps: int = 0,
                     stock_init: bool = False) -> dict:
    """What the policy will actually do, read off the loaded config.

    Read from `policy.config`, never from the CLI defaults: the checkpoint
    ships n_action_steps=64 and every eval here overrides it to 2, so the
    argument and the effective value routinely differ.
    """
    c = policy.config
    return {"n_action_steps": int(c.n_action_steps),
            "num_inference_steps": int(c.num_inference_steps),
            "horizon": int(c.horizon),
            "control_freq": int(control_freq),
            "max_episode_steps": int(max_episode_steps or 0),
            "stock_init": bool(stock_init)}


def report_inference_config(cfg: dict, note: str = "") -> None:
    print("[inference] " + "  ".join(f"{k}={v}" for k, v in cfg.items())
          + (f"\n  -> {note}" if note else ""), flush=True)


def state_scale(preprocessor):
    """Per-dimension std of observation.state, for printing sigma in real units.

    Best effort: the pipeline's shape is lerobot's, not ours. Returns None
    rather than guessing, and the banner then reports normalized units only.
    """
    for step in getattr(preprocessor, "steps", []) or []:
        stats = getattr(step, "stats", None)
        if isinstance(stats, dict) and "observation.state" in stats:
            s = stats["observation.state"]
            for key in ("std", "max"):
                v = s.get(key) if isinstance(s, dict) else getattr(s, key, None)
                if v is not None:
                    return np.asarray(v, dtype=float), key
    return None


def blur_images(batch: dict, factor: int, cams=None) -> int:
    """Destroy detail finer than `factor` pixels, keeping every shape identical.

    Downsample then upsample back, so the ViT still sees the same resolution
    and produces the same token count -- only the information below `factor`
    pixels is gone. That is what makes this a clean test of whether the policy
    USES fine visual detail: an image at a genuinely lower resolution would
    also change the token grid, and then a drop could not be attributed.

    Flat success rate under factor 2 means adding vision capacity (a finer
    DINO path, a larger vision_input_size, LoRA on the tower) is buying
    resolution the policy already declines to use.
    """
    n = 0
    for k in list(batch):
        if not k.startswith("observation.image"):
            continue
        if cams and not any(c in k for c in cams):
            continue
        v = batch[k]
        if not torch.is_tensor(v) or v.dim() < 3:
            continue
        flat = v.reshape(-1, *v.shape[-3:]) if v.dim() > 4 else v
        h, w = flat.shape[-2:]
        small = torch.nn.functional.interpolate(
            flat, size=(max(1, h // factor), max(1, w // factor)),
            mode="area")
        back = torch.nn.functional.interpolate(
            small, size=(h, w), mode="bilinear", align_corners=False)
        batch[k] = back.reshape(v.shape)
        n += 1
    return n


def build_batch(obs_list, tasks, hist: StateHistory, preprocessor, device,
                state_noise: float = 0.0, state_noise_dims=None,
                blur: int = 0, blur_cams=None):
    from lerobot.envs.utils import preprocess_observation

    stacked = {
        "pixels": {cam: np.stack([o["pixels"][cam] for o in obs_list])
                   for cam in obs_list[0]["pixels"]},
        "agent_pos": np.stack([o["agent_pos"] for o in obs_list]),
    }
    batch = preprocess_observation(stacked)
    # Override the single current frame with the full window. preprocess_observation
    # only ever produces (B, D); the model wants (B, T, D) and slices [:, -1] for
    # the state token, exactly as in training.
    batch["observation.state"] = torch.from_numpy(hist.stack()).float()
    batch["task"] = list(tasks)
    # Before the preprocessor, on the raw [0, 1] frames: that is where "detail
    # finer than N pixels" is a statement about the camera rather than about
    # whatever affine the normalizer applies.
    if blur > 1:
        blur_images(batch, blur, blur_cams)
    batch = preprocessor(batch)

    if state_noise > 0.0:
        # AFTER the preprocessor, so sigma is in the same normalized units the
        # sibling trainers use (apply_joint_augmentations: randn * 0.02). That
        # makes this measurement directly answer "what sigma should training
        # use", instead of needing a unit conversion to be trusted.
        #
        # ONE offset for the whole (B, T, D) window, not per frame: the motion
        # encoder reads differences, so independent per-frame noise would inject
        # a velocity spike that the real failure mode -- being a few millimetres
        # off -- does not produce. A constant offset leaves every difference
        # unchanged and moves only the position.
        s = batch["observation.state"]
        off = torch.randn(s.shape[0], 1, s.shape[-1], device=s.device,
                          dtype=s.dtype) * state_noise
        if state_noise_dims is not None:
            keep = torch.zeros(s.shape[-1], device=s.device, dtype=s.dtype)
            keep[list(state_noise_dims)] = 1.0
            off = off * keep
        batch["observation.state"] = s + off.expand_as(s)
    return batch


@torch.no_grad()
class MotionAccumulator:
    """End-effector displacement per policy chunk, split by success.

    The evidence that this family's success rate rides on the per-chunk noise
    re-draw is that --fixed_episode_noise costs 25 points. A 25-point drop from
    freezing x_1 is too large for "one draw was unlucky": with a FIXED draw the
    same observation returns the same action forever, so a policy that walks
    into a bad state never leaves it. That reading says the re-draw is an
    ESCAPE mechanism, not a lottery -- and it predicts something measurable that
    nothing here has ever recorded: failures should contain long stretches where
    the arm is not moving at all.

    So log the distance the end-effector travels between consecutive chunk
    boundaries (state dims 0:3, metres) and report, separately for successful
    and failed episodes:

      median   the typical per-chunk travel
      still%   fraction of chunks under `still_m`
      streak   longest CONSECUTIVE run of still chunks -- the actual "stuck"
               number, and the one a scripted retreat or an adaptive noise
               scale would trigger on

    A failure profile that looks like the success profile means the policy is
    moving the whole time and simply missing: a precision problem, and retreating
    to a home pose would not help. A failure profile with long still streaks
    means it is wedged, and escape is worth engineering.
    """

    def __init__(self, still_m: float = 0.002):
        self.still_m = float(still_m)
        self.ok, self.bad = [], []      # per episode: (median, still_frac, streak, n)

    @staticmethod
    def _summarise(disp, still_m):
        import numpy as _np
        if not disp:
            return None
        d = _np.asarray(disp, dtype=float)
        still = d < still_m
        best = cur = 0
        for v in still:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return float(_np.median(d)), float(still.mean()), int(best), int(d.size)

    def add(self, disp, success: bool):
        r = self._summarise(disp, self.still_m)
        if r is not None:
            (self.ok if success else self.bad).append(r)

    def report(self) -> dict | None:
        import numpy as _np
        if not self.ok and not self.bad:
            return None
        def agg(rows):
            if not rows:
                return None
            a = _np.asarray(rows, dtype=float)
            return {"episodes": int(a.shape[0]),
                    "median_disp_m": round(float(_np.median(a[:, 0])), 5),
                    "still_frac": round(float(_np.mean(a[:, 1])), 4),
                    "max_still_streak_chunks": int(a[:, 2].max()),
                    "mean_still_streak_chunks": round(float(_np.mean(a[:, 2])), 1),
                    "mean_chunks": round(float(_np.mean(a[:, 3])), 1)}
        return {"still_threshold_m": self.still_m,
                "success": agg(self.ok), "failure": agg(self.bad)}


class RoutingAccumulator:
    """Per-denoising-step routing statistics, accumulated over chunks.

    The router runs inside _run_dit, and sample_actions calls that
    num_inference_steps times. Two of the router's four inputs (the time
    embedding and the pooled NOISY action) change at every one of those steps,
    so the expert mixture is re-decided all the way down the ODE. Three things
    follow that nothing in this repo has ever measured:

      SWITCH RATE  does the selected top-k SET change within one chunk? The ODE
                   is then integrating a field whose own definition moves
                   mid-trajectory.
      EARLY->LATE  does the first step (t=1, coarse: WHICH object) prefer
                   different experts from the last (t->0, fine: placement)?
                   That is the axis the disjoint VLM depth bands exist for, and
                   a flat answer means the bands are not being used as designed.
      DRAW SPREAD  how much does routing differ between chunks? action_pool is a
                   quarter of the router's input and is derived from the noise
                   draw, so the per-chunk re-draw ALREADY perturbs expert
                   selection -- this says by how much.
    """

    def __init__(self, top_k: int):
        self.top_k = int(top_k)
        self.n_chunks = 0
        self.n_switch = 0
        self.first = None      # summed weights at t=1
        self.last = None       # summed weights at the final step
        self.mean = None
        self.first_sets, self.last_sets = [], []

    def add(self, trace, live_idx):
        """trace: list of (B, E) tensors, one per denoising step, ODE order."""
        import numpy as _np
        if not trace or not live_idx:
            return
        W = _np.stack([t.numpy() for t in trace], axis=0)   # (steps, B, E)
        W = W[:, live_idx, :]
        S, B, E = W.shape
        # top_k=0 is DENSE: the "selected set" is all E, so switch_rate would be
        # trivially 0 and jaccard trivially 1 -- both uninformative. Fall back to
        # the top half, which keeps the set statistics meaningful and is stated
        # in the report as set_k.
        k = self.top_k if 0 < self.top_k < E else max(1, E // 2)
        self.set_k = k
        if self.first is None:
            self.first = _np.zeros(E); self.last = _np.zeros(E); self.mean = _np.zeros(E)
        self.first += W[0].sum(0); self.last += W[-1].sum(0); self.mean += W.sum((0, 1)) / S
        sel = _np.argsort(-W, axis=-1)[:, :, :k]            # (steps, B, k)
        for b in range(B):
            sets = [frozenset(sel[s, b].tolist()) for s in range(S)]
            self.n_chunks += 1
            self.n_switch += int(len(set(sets)) > 1)
            self.first_sets.append(sets[0]); self.last_sets.append(sets[-1])

    def report(self) -> dict | None:
        if not self.n_chunks or self.first is None:
            return None
        import numpy as _np
        n = self.n_chunks
        f, l, m = self.first / n, self.last / n, self.mean / n
        jac = [len(a & b) / max(len(a | b), 1) for a, b in zip(self.first_sets, self.last_sets)]
        return {
            "chunks": n,
            "set_k": getattr(self, "set_k", self.top_k),
            "dense": self.top_k == 0,
            "switch_rate": self.n_switch / n,
            "first_step_weights": [round(float(x), 4) for x in f],
            "last_step_weights": [round(float(x), 4) for x in l],
            "mean_weights": [round(float(x), 4) for x in m],
            "early_late_L1": round(float(_np.abs(f - l).sum()), 4),
            "early_late_set_jaccard": round(float(_np.mean(jac)), 4),
        }


def eval_task(policy, preprocessor, postprocessor, suite, suite_name: str,
              task_id: int, episodes: int, num_envs: int, device: str,
              max_episode_steps: int, seed: int, expected_cams: list[str],
              policy_seed: int | None = None,
              video_cb=None, videos_per_task: int = 0, heartbeat: int = 50,
              instruction: str | None = None, state_noise: float = 0.0,
              state_noise_dims=None, blur: int = 0, blur_cams=None,
              history_mode: str = "real", routing_acc=None, motion_acc=None,
              action_offset=None, envs=None, init_state_offset: int = 0,
              init_state_stride: int = 1):
    """-> (n_success, n_episodes, mean_success_steps, n_chunks, task_description,
    per_episode_success).

    The per-episode vector is what makes two checkpoints COMPARABLE. Episode i
    starts from the same canonical init state in every run (fixed_init_states),
    so two evals are a PAIRED sample and McNemar applies. Comparing only the
    two rates throws that away and leaves ~15pp of unpaired noise at n=20 --
    enough to invent a 20-point "regression" between adjacent checkpoints.
    """
    from lerobot.envs.libero import LiberoEnv

    # Re-seed PER TASK, not once per process. The policy draws its flow noise
    # from the global torch RNG, which advances with every chunk, so a run of
    # `--task_ids 4 0 5` reached task 0 with thousands of draws already spent
    # and gave it a different noise stream than a run of `--task_ids 0 ...`.
    # That is not a subtle effect: it produced task 0 at 45% in one ordering
    # and 85% in another, on the SAME checkpoint -- a 40-point artefact that
    # reads as a result.
    #
    # Keyed on task_id so ordering, and which tasks are in the run at all,
    # cannot reach the noise. Two runs are then paired on BOTH the layout and
    # the noise: episode i sees the same x_1 sequence in both, and diverges
    # only where the policy itself does.
    #
    # `policy_seed` is separate from `seed` on purpose: `seed` picks the
    # LAYOUTS (env.reset below) and policy_seed picks the flow noise. Holding
    # the first and moving the second is the null control for every A/B run
    # through this script -- how many episodes change outcome when NOTHING
    # about the policy or its inputs changed, only the sampled x_1. Without
    # that number, "state noise flipped 31 of 60 episodes" cannot be told
    # apart from "this policy flips 31 of 60 episodes on its own".
    ps = seed if policy_seed is None else policy_seed
    torch.manual_seed(ps + task_id)
    np.random.seed((ps + task_id) % (2 ** 32))

    # Building an OffScreenRenderEnv takes seconds and there are num_envs of
    # them PER TASK (LiberoEnv binds its bddl file at construction, so they
    # cannot be reused across tasks). Say so: this is minutes of silence
    # before a single rollout step happens.
    # A caller that evaluates the SAME task many times -- golden-ticket search
    # runs hundreds of rollout sets per task -- passes its own envs in and pays
    # the build once instead of once per call.
    own_envs = envs is None
    if own_envs:
        t_build = time.time()
        print(f"  task {task_id:2d}: building {num_envs} envs...", end="", flush=True)
        envs = [LiberoEnv(task_suite=suite, task_id=task_id,
                          task_suite_name=suite_name, obs_type="pixels_agent_pos",
                          init_states=True, episode_index=0)
                for _ in range(num_envs)]
        print(f" {time.time() - t_build:.0f}s", flush=True)
    try:
        probe, _ = envs[0].reset(seed=seed)
        got = sorted(probe["pixels"].keys())
        want = [c.split(".")[-1] for c in expected_cams]
        missing = [c for c in want if c not in got]
        if missing:
            raise SystemExit(
                f"LIBERO provides cameras {got}; the policy expects {want} "
                f"(the config's own camera list: {expected_cams}). "
                f"_encode_images drops "
                f"missing cameras SILENTLY, so this would score a "
                f"differently-conditioned model. Pass a camera_name_mapping to "
                f"LiberoEnv for this lerobot version.")

        task_desc = envs[0].task_description
        n_states = len(envs[0]._init_states)
        horizon_cap = min(envs[0]._max_episode_steps, max_episode_steps) \
            if max_episode_steps else envs[0]._max_episode_steps
        # Seeded off the task, like the policy noise, so `shuffled` is a
        # reproducible intervention rather than a fresh coin every run.
        hist = StateHistory(num_envs, policy.config.n_obs_steps,
                            mode=history_mode, seed=ps + task_id)

        autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device == "cuda"
                    else torch.autocast(device_type="cpu", enabled=False))

        n_batches = (episodes + num_envs - 1) // num_envs
        # Never truncate the instruction. LIBERO's tasks share a prefix and
        # differ at the END ("...between the plate and the ramekin"), so
        # clipping the tail hides the only part that distinguishes them -- and
        # makes a display limit look like the tokenizer dropping text.
        # The success criterion always comes from the env, i.e. from the REAL
        # task. Only what the policy is told changes.
        told = instruction if instruction is not None else task_desc
        print(f"    {episodes} episodes in {n_batches} batch(es) of {num_envs}, "
              f"cap {horizon_cap} steps\n    task: {task_desc!r}", flush=True)
        if instruction is not None:
            print(f"    ABLATED, policy is told: {told!r}", flush=True)

        successes, steps_to_success, n_chunks = [], [], 0
        for start in range(0, episodes, num_envs):
            t_batch = time.time()
            n_live = min(num_envs, episodes - start)
            policy.reset()
            obs_list, frames = [], [[] for _ in range(num_envs)]
            for i in range(num_envs):
                # Episode index -> init state. Explicit, not seed-derived: the
                # canonical set is 50 layouts and coverage should be exact and
                # reproducible, not a hash of the seed.
                # init_state_offset separates SEARCH layouts from EVAL
                # layouts. A standard 20-episode eval uses ids 0-19 of the
                # canonical 50, so searching at offset 20 leaves the reported
                # numbers uncontaminated and directly comparable to every row
                # in the tracker. Searching and reporting on the same layouts
                # would measure how well a ticket was fitted, not how well it
                # generalises.
                # stride 0 puts EVERY env on the same layout, which is how
                # ticket search scores N candidates against one another: the
                # tickets differ per env, the problem does not. With stride 1
                # this is the ordinary episode -> layout mapping.
                envs[i]._init_state_id = (
                    init_state_offset + (start + i) * init_state_stride) % n_states
                o, _ = envs[i].reset(seed=seed + start + i)
                obs_list.append(o)
                hist.reset(i, o["agent_pos"])

            done = [i >= n_live for i in range(num_envs)]   # pad slots start done
            succ = [False] * num_envs
            _mo_prev = [None] * num_envs
            _mo_disp = [[] for _ in range(num_envs)]
            steps = [0] * num_envs
            t = 0
            while not all(done) and t < horizon_cap:
                # The batch dim stays at num_envs even as envs finish: the action
                # queue inside select_action is keyed on batch size, and resizing
                # it mid-chunk would drop the actions the live envs still owe.
                batch = build_batch(obs_list, [told] * num_envs, hist,
                                    preprocessor, device,
                                    state_noise, state_noise_dims,
                                    blur, blur_cams)
                # An empty queue means this call will run the prefix. Counting
                # it here rather than after the call keeps it correct at
                # n_action_steps=1, where the queue is empty again on return.
                drew_chunk = getattr(policy, "_drew_chunk", None)
                if drew_chunk is None:
                    drew_chunk = not policy._action_queue
                with autocast:
                    action = policy.select_action(batch)
                n_chunks += int(drew_chunk)
                if drew_chunk and motion_acc is not None:
                    for i in range(num_envs):
                        if done[i]:
                            continue
                        _p = np.asarray(obs_list[i]["agent_pos"], dtype=float).reshape(-1)[:3]
                        if _mo_prev[i] is not None:
                            _mo_disp[i].append(float(np.linalg.norm(_p - _mo_prev[i])))
                        _mo_prev[i] = _p
                if drew_chunk and routing_acc is not None:
                    tr = getattr(policy.model, "_routing_trace", None)
                    if tr:
                        routing_acc.add(tr, [i for i in range(num_envs) if not done[i]])
                env_action = postprocessor(action.float().cpu()).numpy()
                if action_offset:
                    # Applied AFTER the postprocessor, so the units are the
                    # controller's own [-1, 1] and the value is exactly what the
                    # env receives (before its clip).
                    #
                    # NOTE the action is an OSC DELTA. A constant offset is a
                    # constant VELOCITY bias, not a fixed position shift, and the
                    # policy is closed-loop: it sees the drift and commands
                    # against it, so the steady-state height change is smaller
                    # than offset x output_max and depends on the policy's own
                    # feedback. Sweep it and read the success rate; do not try to
                    # predict the millimetres.
                    for _d, _v in action_offset:
                        env_action[:, _d] += _v

                for i in range(num_envs):
                    if done[i]:
                        continue
                    lo, hi = envs[i].action_space.low, envs[i].action_space.high
                    a = np.clip(env_action[i], lo, hi).astype(np.float32)
                    o, _r, terminated, truncated, info = envs[i].step(a)
                    obs_list[i] = o
                    hist.push(i, o["agent_pos"])
                    steps[i] += 1
                    # Only the first few env slots buffer frames. A full 256x256
                    # episode is ~100 MB; recording all of them would cost more
                    # RAM than the policy does, to write videos we then discard.
                    if video_cb is not None and i < videos_per_task:
                        frames[i].append(o["pixels"][got[0]])
                    if terminated or truncated:
                        # LiberoEnv auto-resets on termination, so the env must
                        # not be stepped again or it silently starts a new
                        # episode and pollutes the next chunk's observation.
                        done[i] = True
                        succ[i] = bool(info.get("is_success", False))
                        if motion_acc is not None:
                            motion_acc.add(_mo_disp[i], succ[i])
                t += 1

                # Heartbeat. Without it a batch that runs to the cap is many
                # minutes of total silence, and there is no way to tell a slow
                # rollout from a hung one. `live` falling to 0 early means the
                # episodes are terminating; live staying at num_envs to the cap
                # means everything is timing out, i.e. failing.
                if heartbeat and t % heartbeat == 0:
                    live = sum(1 for i in range(n_live) if not done[i])
                    hit = sum(succ[:n_live])
                    el = time.time() - t_batch
                    print(f"      t={t:4d}/{horizon_cap}  live={live}/{n_live}  "
                          f"success={hit}  {el:5.0f}s  "
                          f"({el / max(t, 1) * horizon_cap:.0f}s if it runs to cap)",
                          flush=True)

            if motion_acc is not None:
                # Episodes that ran to horizon_cap never reach the terminated /
                # truncated branch, so done[i] stays False and they were never
                # flushed. Those are exactly the failures this measurement is
                # for -- on libero_10 T8 the whole batch runs to the cap -- so
                # dropping them would have measured only the episodes that
                # ENDED, i.e. overwhelmingly the successes.
                for i in range(n_live):
                    if not done[i]:
                        motion_acc.add(_mo_disp[i], False)

            hit = sum(succ[:n_live])
            print(f"    batch {start // num_envs + 1}/{n_batches}: "
                  f"{hit}/{n_live} success in {t} steps, "
                  f"{time.time() - t_batch:.0f}s", flush=True)

            for i in range(n_live):
                successes.append(succ[i])
                if succ[i]:
                    steps_to_success.append(steps[i])
                if video_cb is not None:
                    video_cb(task_id, start + i, frames[i], success=bool(succ[i]))

        mean_steps = float(np.mean(steps_to_success)) if steps_to_success else float("nan")
        return (sum(successes), len(successes), mean_steps, n_chunks, task_desc,
                [int(s) for s in successes])
    finally:
        if own_envs:
            for e in envs:
                try:
                    e.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
def make_video_writer(video_dir: Path | None, max_per_task: int,
                      which: str = "fail"):
    """Record rollouts. `which` selects fail / success / both.

    Failures alone answer "is it broken"; telling WHY usually needs the pair,
    because the question is what the successful runs do differently. The
    per-outcome caps are separate so asking for both does not let whichever
    outcome is more common crowd the other out of the quota.
    """
    if video_dir is None:
        return None
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[eval] --video_dir set but imageio is not installed; skipping video.")
        return None
    video_dir.mkdir(parents=True, exist_ok=True)
    written: dict = {}

    def cb(task_id, episode, frames, success=False):
        if not frames:
            return
        if which == "fail" and success:
            return
        if which == "success" and not success:
            return
        key = (task_id, bool(success))
        if written.get(key, 0) >= max_per_task:
            return
        written[key] = written.get(key, 0) + 1
        tag = "OK" if success else "FAIL"
        path = video_dir / f"task{task_id:02d}_ep{episode:03d}_{tag}.mp4"
        imageio.mimwrite(path, [np.asarray(f) for f in frames], fps=10)

    return cb


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suites", nargs="+",
                   default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    p.add_argument("--noise_ticket", default=None,
                   help=".npy of shape (horizon, action_dim) used as the "
                        "CONSTANT initial noise for every action step and "
                        "every episode, instead of drawing x_1 ~ N(0,I) -- the "
                        "golden ticket of Patil et al. 2026. Training is "
                        "untouched; this only changes sampling, so it applies "
                        "to a frozen checkpoint. Produced by "
                        "search_golden_ticket.py.")
    p.add_argument("--noise_tickets", default=None,
                   help="A golden_tickets.safetensors bundle, or 'auto' to "
                        "look for one in the checkpoint directory. Each task "
                        "is evaluated with ITS OWN ticket; tasks the bundle "
                        "does not cover fall back to Gaussian sampling, and "
                        "which ones did what is recorded in the result JSON "
                        "-- a suite average that silently mixes the two is "
                        "not comparable to anything.")
    p.add_argument("--use_weak_tickets", action="store_true",
                   help="Use tickets the search marked beats_baseline=false, "
                        "i.e. ones that did not beat Gaussian at all. "
                        "Off by default: a ticket that only MATCHES Gaussian "
                        "is strictly worse than not using one, because it also "
                        "removes the per-chunk re-draw this benchmark measured "
                        "at 25 points. Those tasks fall back to Gaussian and "
                        "say so.")
    p.add_argument("--init_state_offset", type=int, default=0,
                   help="Shift which of the canonical 50 layouts the episodes "
                        "use. A standard 20-episode eval takes ids 0-19, so a "
                        "ticket SEARCHED at offset 20 can be REPORTED at "
                        "offset 0 without having been fitted to the layouts it "
                        "is scored on. Leave at 0 for anything reportable.")
    p.add_argument("--task_ids", nargs="+", type=int, default=None,
                   help="Default: every task in each suite.")
    p.add_argument("--episodes", type=int, default=50,
                   help="Per task. 50 is the canonical LIBERO count and matches "
                        "the number of init states, so each is visited once.")
    p.add_argument("--num_envs", type=int, default=10,
                   help="Envs stepped in lockstep in THIS process, batched "
                        "through one policy forward. MuJoCo's EGL context is "
                        "per-thread, so these must not be threaded; sequential "
                        "stepping in one process is correct and the policy "
                        "forward (the expensive part) still batches.")
    p.add_argument("--control_freq", type=int, default=10,
                   help="MUST be 10 to match the dataset. Changing it invalidates "
                        "comparison with every other number in this repo.")
    p.add_argument("--max_episode_steps", type=int, default=0,
                   help="0 = the suite default.")
    p.add_argument("--num_inference_steps", type=int, default=0,
                   help="0 = the checkpoint's config (4). Re-run at 16 to test "
                        "whether the shortcut term made few-step inference valid.")
    p.add_argument("--noise_cycle", default=None,
                   help="An (m, horizon, action_dim) .npy used one vector per "
                        "chunk in turn, so chunk k takes t[k %% m]. m=1 is a "
                        "plain ticket and large m of random vectors "
                        "approximates ordinary sampling, which makes m the "
                        "dial that separates two explanations for why a "
                        "frozen x_1 costs long T0 about 50 points while "
                        "costing the short suites little: the vector, or the "
                        "fact that 79 consecutive chunks share it. Training "
                        "never showed the model two chunks with the same x_1. "
                        "Still deterministic, so the same layout replays bit "
                        "for bit.")
    p.add_argument("--n_action_steps", type=int, default=0,
                   help="0 = the checkpoint's config (8). Steps of each chunk "
                        "executed open-loop before replanning. At 10 Hz, 8 is "
                        "0.8 s and ~35 replans per episode; wiltechs_vla and "
                        "wiltechs_moe run 32 of a 64 horizon, so they "
                        "re-decide 4x less often. Each replan redraws the "
                        "noise, i.e. resamples WHICH plan to follow, so a high "
                        "replan rate is a candidate cause of the stumbling "
                        "approach. Cannot exceed the trained horizon.")
    p.add_argument("--fixed_episode_noise", action="store_true",
                   help="Draw x_1 once per episode and reuse it for every "
                        "replan. The integration is deterministic given the "
                        "noise, so this keeps the policy on ONE branch of a "
                        "multimodal action distribution while staying fully "
                        "reactive to the observation. A bad branch now costs "
                        "the whole episode instead of 0.8 s, so read the "
                        "success distribution, not only the mean.")
    p.add_argument("--stock_init", action="store_true",
                   help="Disable the init-state ordering fix. For an A/B against "
                        "the canonical 50 layouts; not for reportable numbers.")
    p.add_argument("--dataset_id", default=None,
                   help="Only for checkpoints saved without their processors.")
    p.add_argument("--device", default=None)
    p.add_argument("--render_gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--sample_noise_scale", type=float, default=None,
                   help="Temperature on the initial flow noise x_1 (trained "
                        "value 1.0). Below 1 pulls every sample toward the "
                        "centre of the action distribution; 0 makes the policy "
                        "deterministic. Distinct from --fixed_episode_noise, "
                        "which commits to one RANDOM draw rather than moving "
                        "toward the middle.")
    p.add_argument("--policy_seed", type=int, default=None,
                   help="Seed for the POLICY's flow noise, separate from "
                        "--seed which picks the layouts. Defaults to --seed. "
                        "Re-running with only this changed is the null control: "
                        "same checkpoint, same layouts, same inputs, different "
                        "x_1. However many episodes flip is the floor that any "
                        "--state_noise or --image_blur delta has to clear.")
    p.add_argument("--video_dir", default="failed_rollouts", type=Path,
                   help="Write up to --videos_per_task FAILED episodes per task. "
                        "This repo's grasp-vs-selection diagnoses came from "
                        "watching these, not from the success rate.")
    p.add_argument("--videos_per_task", type=int, default=2,
                   help="Cap per task AND per outcome, so --videos_of both "
                        "gives up to this many of each rather than letting the "
                        "commoner outcome take the whole quota.")
    p.add_argument("--videos_of", choices=("fail", "success", "both"),
                   default="fail",
                   help="Which rollouts to record (default: fail). Failures "
                        "alone answer 'is it broken'; saying WHY usually needs "
                        "'both', since the question is what the successful runs "
                        "do differently.")
    p.add_argument("--ablate_lang", action="store_true",
                   help="Tell the policy ANOTHER task's instruction while "
                        "scoring against the real one. The bridge between "
                        "'the CE depends on language a little' and 'behaviour "
                        "depends on language': if the success rate does not "
                        "move, the instruction is not driving the policy and "
                        "no amount of further training changes that. Run it "
                        "against an identical non-ablated run -- same "
                        "checkpoint, seed, episodes and cap.")
    p.add_argument("--instruction_override", default=None,
                   help="Tell the policy this exact instruction instead of the "
                        "task's own, while still scoring against the real task. "
                        "For a SPECIFIC confusion, where --ablate_lang's fixed "
                        "half-suite offset does not put the two tasks in "
                        "question against each other. Also takes a rephrasing, "
                        "to ask whether a more distinctive wording is followed.")
    p.add_argument("--instruction_from_task", type=int, default=None,
                   help="Same, but pulls the instruction off another task in "
                        "this suite by id -- no chance of a typo silently "
                        "testing a different sentence. Use with --task_ids to "
                        "swap a pair: --task_ids 7 --instruction_from_task 9.")
    p.add_argument("--allow_missing_weights", action="store_true",
                   help="Score a checkpoint that does not supply every tensor "
                        "the current code declares. Refused by default when any "
                        "of them initialises NONZERO, because then the loaded "
                        "model is not the one that was trained and the number "
                        "compares to nothing. Zero-init tensors contribute "
                        "nothing and are allowed without this.")
    p.add_argument("--history_mode", default="real",
                   choices=("real", "frozen", "shuffled", "noise"),
                   help="Ablate the observation.state WINDOW the motion-vector "
                        "encoder reads. 'shuffled' permutes the T frames per "
                        "call: every marginal is preserved and only the "
                        "temporal order dies, so a drop cannot be blamed on "
                        "unfamiliar values -- it is the control for "
                        "ARCHITECTURE.md 8.2 (does this policy ride the "
                        "demonstrator's momentum instead of the image?). "
                        "'frozen' repeats the newest frame, which is what "
                        "every episode's first step already looks like. "
                        "Read 'shuffled'; 'frozen' alone is ambiguous.")
    p.add_argument("--action_offset", nargs=2, action="append", default=None,
                   metavar=("DIM", "VALUE"),
                   help="Add a constant to one action dim at inference, in the "
                        "controller's own [-1, 1] units, every step. Repeatable: "
                        "--action_offset 2 0.04 --action_offset 0 -0.02. This is "
                        "the cheap test for whether a placement error is BIAS or "
                        "VARIANCE: if a small +z offset moves the success rate on "
                        "the tasks that 'get near but land a little low', the "
                        "residual is a systematic offset -- which RL can learn "
                        "away as a constant, and which behaviour cloning cannot, "
                        "because the demos never contain 'you are 5 mm low, go "
                        "up'. If nothing moves, it is variance and the lever is "
                        "perception, not RL. The action is an OSC DELTA, so a "
                        "constant here is a VELOCITY bias that the closed loop "
                        "partially rejects: sweep small values (0.02-0.10) and "
                        "read the success rate rather than predicting the "
                        "millimetres.")
    p.add_argument("--log_motion", action="store_true",
                   help="Record how far the end-effector travels between "
                        "consecutive policy chunks (state dims 0:3), reported "
                        "separately for successful and failed episodes. This is "
                        "the measurement that decides what 'stuck' means here: "
                        "--fixed_episode_noise costs 25 points, which is too "
                        "large for bad luck on one draw and reads instead as "
                        "'a frozen draw returns the same action forever, so a "
                        "policy that walks into a bad state never leaves it'. "
                        "If that is right, failures carry long motionless "
                        "streaks and escape is worth engineering; if failures "
                        "move as much as successes, the arm is missing rather "
                        "than wedged and a retreat would not help.")
    p.add_argument("--still_threshold", type=float, default=0.002,
                   help="Metres of end-effector travel between chunk boundaries "
                        "below which a chunk counts as motionless (default "
                        "0.002 = 2 mm). For scale: one control step can command "
                        "at most 0.9375 x 0.05 m = 47 mm, and typical demo "
                        "motion is ~17-22 mm per step.")
    p.add_argument("--router_top_k", type=int, default=None,
                   help="Override the checkpoint's router_top_k at inference "
                        "(wilro_moe). 0 = dense, all experts weighted. The "
                        "model trained with its own k AND with N(0, 0.5) "
                        "exploration noise on the router logits that inference "
                        "does not have, so it already runs a routing "
                        "distribution it never saw; changing k widens that "
                        "deliberately. Nothing breaks -- the mask is applied "
                        "then renormalised -- but the mixture is not the one "
                        "the loss was minimised over. Raising k averages in the "
                        "experts the router ranked WORST for this sample, which "
                        "lowers output variance: that helps MSE and may hurt SR, "
                        "since this family's success rate rides on the per-chunk "
                        "re-draw. Measure, do not predict.")
    p.add_argument("--log_routing", action="store_true",
                   help="Record which experts the router picks at EACH "
                        "denoising step (it re-decides at every one -- two of "
                        "its four inputs change with t and with the noisy "
                        "action). Reports the within-chunk switch rate and the "
                        "first-step vs last-step weights, i.e. whether the "
                        "disjoint VLM depth bands are actually specialising "
                        "coarse-vs-fine. Writes a 'routing' block to the JSON.")
    p.add_argument("--state_noise", type=float, default=0.0,
                   help="Gaussian offset added to observation.state, sigma in "
                        "NORMALIZED units -- the same units the sibling trainers "
                        "augment in (train_wiltechs_moe uses 0.02). Sweep it to "
                        "find out whether the policy is brittle to being a few "
                        "millimetres off, which is what a fumbled grasp is. Flat "
                        "SR means state augmentation buys nothing; a collapse "
                        "means it does, and the collapse point is the sigma to "
                        "train at. One offset per episode-window, so the motion "
                        "differences are untouched.")
    p.add_argument("--state_noise_dims", nargs="+", type=int, default=None,
                   help="Restrict --state_noise to these state indices, e.g. "
                        "0 1 2 for end-effector position only. Default: all "
                        "dims, matching the sibling trainers.")
    p.add_argument("--image_blur", type=int, default=0,
                   help="Downsample the camera frames by this factor and "
                        "upsample back, destroying detail finer than N pixels "
                        "while keeping the token grid identical. The mirror of "
                        "--state_noise: flat SR under factor 2 means the policy "
                        "does not use fine visual detail, so a finer DINO path, "
                        "a larger --vision_input_size, or LoRA on the vision "
                        "tower is buying resolution it declines to read.")
    p.add_argument("--vision_input_size", type=int, default=0,
                   help="Override the resolution the FROZEN VLM tower reads. "
                        "SmolVLM2-500M is pretrained at 512; the shipped "
                        "configs run it at 384, i.e. BELOW native, at 36 tokens "
                        "per camera instead of 64. No retraining is needed to "
                        "try it, but read it asymmetrically: a gain is "
                        "evidence, a loss is not, since the projection and the "
                        "experts were fitted to the old token count. Gate this "
                        "behind --image_blur 2 -- if SR is flat under blur the "
                        "policy declines to read fine detail and more "
                        "resolution cannot help.")
    p.add_argument("--temporal_ensemble_coeff", type=float, default=None,
                   help="Average every chunk still covering the current "
                        "timestep, weighted exp(-coeff x age_in_steps). COSTS "
                        "NO EXTRA FORWARD PASSES -- the draw cadence stays "
                        "n_action_steps; horizon 64 / n_action_steps 2 means 32 "
                        "chunks already predict each step and 31 are discarded. "
                        "0 = off (bit-identical to the old path); a large coeff "
                        "also reduces to it. Try 0.01 (near-uniform over the "
                        "window) and 0.1 (K_eff ~ 10, fresher). Expect the "
                        "DEEPEST stalls to get worse -- every variance "
                        "reduction in this project did -- and pair it with "
                        "--stall_noise_scale.")
    p.add_argument("--stall_noise_scale", type=float, default=None,
                   help="Noise scale used for the envs that have stopped "
                        "moving, and only those. 0 = off. UNTESTED: it is a "
                        "hypothesis from the motion column (noise is what "
                        "escapes a stall), not a measured result.")
    p.add_argument("--stall_rel_threshold", type=float, default=None,
                   help="'Still' = state step below this fraction of the "
                        "episode's own largest step. Relative, so it needs no "
                        "units. Default 0.1.")
    p.add_argument("--stall_patience", type=int, default=None,
                   help="Consecutive still chunks before the stall scale "
                        "fires. Default 5.")
    p.add_argument("--image_blur_cams", nargs="+", default=None,
                   help="Restrict --image_blur to camera keys containing these "
                        "substrings, e.g. image2 for the wrist view alone. "
                        "Default: every camera.")
    p.add_argument("--heartbeat", type=int, default=50,
                   help="Env steps between progress lines inside a rollout. A "
                        "batch that runs to the episode cap is minutes of "
                        "silence otherwise, with no way to tell slow from hung. "
                        "0 = off.")
    p.add_argument("--out", default=None, help="JSON results path.")
    a = p.parse_args()

    device = a.device or pick_device()
    from checkpoint_utils import resolve_checkpoint

    # --seed used to reach only env.reset(). The POLICY is stochastic -- flow
    # matching draws x_1 fresh for every chunk, which at n_action_steps=2 is
    # 140 draws per episode -- so two runs of the identical command walked
    # different trajectories through identical layouts. That also made the
    # --ablate_lang instruction to use "the same checkpoint, seed, episodes and
    # cap" impossible to honour.
    #
    # Seeding does NOT shrink the error on a single estimate: at n=20 the
    # binomial SE near p=0.85 is ~8 points, which is what an 80% and a 90% run
    # of the same command actually differ by. What it buys is a PAIRED A/B --
    # same layouts and same noise, only the setting under test moving.
    #
    # Pairing is exact only when the two arms draw the same number of samples.
    # Comparing n_action_steps settings changes that count, so the noise
    # streams diverge after the first chunk; --fixed_episode_noise draws once
    # per episode and pairs across those too.
    #
    # This seeds the setup; eval_task re-seeds per task so that TASK ORDER
    # cannot reach the noise. See the comment there.
    torch.manual_seed(a.seed)
    np.random.seed(a.seed % (2 ** 32))

    n_lang = sum(x is not None and x is not False
                 for x in (a.ablate_lang or None, a.instruction_override,
                           a.instruction_from_task))
    if n_lang > 1:
        raise SystemExit(
            "--ablate_lang, --instruction_override and --instruction_from_task "
            "all replace what the policy is told. Pick one; combining them "
            "would report a number nobody could attribute.")

    # Every comparison in the tracker is paired and pins eval_commit; nothing
    # pinned the environment, and the GPU alone decides the kernels' reduction
    # order. Two results with different digests are not strictly paired.
    print(fingerprint_line(), flush=True)

    ckpt = resolve_checkpoint(a.checkpoint, for_resume=False)

    setup_libero_env(a.control_freq, a.render_gpu, a.stock_init)

    policy = load_policy(ckpt, device, a.num_inference_steps,
                         a.n_action_steps, a.fixed_episode_noise,
                         a.sample_noise_scale,
                         router_top_k=a.router_top_k,
                         log_routing=a.log_routing,
                         vision_input_size=a.vision_input_size,
                         temporal_ensemble_coeff=a.temporal_ensemble_coeff,
                         stall_noise_scale=a.stall_noise_scale,
                         stall_rel_threshold=a.stall_rel_threshold,
                         stall_patience=a.stall_patience)
    routing_acc = (RoutingAccumulator(int(getattr(policy.config, "router_top_k", 0) or 0))
                   if a.log_routing else None)
    motion_acc = MotionAccumulator(a.still_threshold) if a.log_motion else None
    action_offset = None
    if a.action_offset:
        action_dim = int(policy.config.action_dim)
        action_offset = []
        for _d, _v in a.action_offset:
            _d, _v = int(_d), float(_v)
            if not 0 <= _d < action_dim:
                raise SystemExit(f"--action_offset dim {_d} outside [0, {action_dim})")
            action_offset.append((_d, _v))
        print("[eval] action offsets (controller units, applied every step, "
              "BEFORE the env clip):")
        for _d, _v in action_offset:
            print(f"    dim {_d}: {_v:+.4f}"
                  + (f"   ~= {_v * 50:+.1f} mm/step commanded, if robosuite's OSC "
                     f"output_max is 0.05 m (UNVERIFIED -- check the LIBERO "
                     f"controller config)" if _d < 3 else ""))
        print("    the action is a DELTA, so this is a velocity bias the "
              "closed loop partially rejects; read SR, not millimetres.")
    report_new_config_fields(policy.config, ckpt)
    report_missing_weights(policy, ckpt, a.allow_missing_weights)
    pre, post = load_processors(ckpt, device, a.dataset_id)
    # The same line the ticket search prints, from the same function,
    # so a mismatch between the two is visible by reading two logs
    # side by side.
    report_inference_config(inference_config(
        policy, a.control_freq, a.max_episode_steps, a.stock_init))
    # THE SILENT DEFAULT THAT HAS COST THIS PROJECT THREE RESULTS. The
    # checkpoint ships n_action_steps=64 and every reported eval passes 2, so
    # --n_action_steps left at 0 runs a policy that replans a handful of times
    # per episode instead of every two steps. It does not error, it does not
    # look wrong in the log, it just returns zero: an 0/50 ticket search, an
    # 0/200 RFT collect, and an 0/20 random-ticket control that read as
    # "tickets do not work on libero_10" until the field was checked.
    if a.n_action_steps == 0 and int(policy.config.n_action_steps) > 8:
        print(f"\n*** WARNING: n_action_steps={int(policy.config.n_action_steps)}, "
              f"the checkpoint's own value, because --n_action_steps was not\n"
              f"    passed. Every reported eval in this project uses 2. At "
              f"{int(policy.config.n_action_steps)} the policy replans a few\n"
              f"    times per episode and the usual result is 0%. Pass "
              f"--n_action_steps 2 unless you mean this.\n", flush=True)
    if a.noise_cycle:
        _cy = np.load(a.noise_cycle)
        _want = (int(policy.config.horizon), int(policy.config.action_dim))
        if _cy.ndim != 3 or tuple(_cy.shape[1:]) != _want:
            raise SystemExit(
                f"--noise_cycle has shape {tuple(_cy.shape)}; it must be "
                f"(m, {_want[0]}, {_want[1]}) -- m vectors of horizon x "
                f"action_dim, used one per chunk in turn.")
        policy.model._noise_cycle = torch.from_numpy(
            _cy.astype(np.float32)).to(device)
        policy.model._noise_cycle_k = 0
        print(f"[cycle] {_cy.shape[0]} vectors, chunk k uses t[k % "
              f"{_cy.shape[0]}]. Deterministic -- the same layout replays "
              f"identically -- but consecutive chunks differ, which a single "
              f"ticket cannot do.", flush=True)
    if a.noise_ticket:
        _tk = np.load(a.noise_ticket)
        _want = (int(policy.config.horizon), int(policy.config.action_dim))
        if _tk.ndim == 3 and tuple(_tk.shape[1:]) == _want:
            # An m-tuple saved by a --cycle search. Route it to the cycling
            # path rather than rejecting it; the shapes are unambiguous.
            policy.model._noise_cycle = torch.from_numpy(
                _tk.astype(np.float32)).to(device)
            policy.model._noise_cycle_k = 0
            print(f"[cycle] --noise_ticket holds {_tk.shape[0]} vectors; "
                  f"using them as a cycle", flush=True)
            a.noise_ticket = None
        elif tuple(_tk.shape) != _want:
            raise SystemExit(
                f"--noise_ticket has shape {tuple(_tk.shape)} but this policy "
                f"needs {_want} (horizon x action_dim), or (m, {_want[0]}, "
                f"{_want[1]}) for a cycle. A ticket is bound to the horizon it "
                f"was searched at.")
        policy.model._noise_ticket = torch.from_numpy(_tk).float().to(device)
        print(f"[ticket] {a.noise_ticket}  shape {_want}  "
              f"norm {float(np.linalg.norm(_tk)):.2f} "
              f"(a N(0,I) draw of this size averages "
              f"{np.sqrt(_want[0] * _want[1]):.1f})")
    _tickets, _tmeta = {}, {}
    if a.noise_tickets:
        import ticket_bundle as tb
        _tickets, _tmeta = tb.load_bundle(ckpt if a.noise_tickets == "auto"
                                          else a.noise_tickets)
        _want = (int(policy.config.horizon), int(policy.config.action_dim))
        # (H, D) is one constant vector; (m, H, D) is a CYCLE, chunk k
        # taking element k mod m, which --cycle searches and which a
        # hand-placed 4-tuple uses. Both live in the same bundle under the
        # same key, and the shapes tell them apart, so rejecting the second
        # was just the single-vector path never having been taught about the
        # other. goal T9 sat at 55% behind this error while the 4-cycle that
        # scores 70% was already in the bundle.
        bad = {k: tuple(v.shape) for k, v in _tickets.items()
               if tuple(v.shape) != _want
               and not (v.ndim == 3 and tuple(v.shape[1:]) == _want)}
        if bad:
            raise SystemExit(
                f"bundle holds tickets of shape {sorted(set(bad.values()))} but "
                f"this policy needs {_want}, or (m, {_want[0]}, {_want[1]}) for "
                f"a cycle; a ticket is bound to the horizon it was searched "
                f"at. Offending keys: {sorted(bad)[:5]}")
        _cyc_keys = sorted(k for k, v in _tickets.items() if v.ndim == 3)
        if _cyc_keys:
            print(f"[cycle] {len(_cyc_keys)} of these are cycles: "
                  + ", ".join(f"{k} (m={_tickets[k].shape[0]})"
                              for k in _cyc_keys))
        print(f"[tickets] {len(_tickets)} in bundle: {sorted(_tickets)}")
        # A ticket is only valid for the inference config it was searched
        # under. n_action_steps is the one that bites: the checkpoint says 64
        # and every eval here passes 2, so a search that forgot the override
        # optimised a policy that replans twice an episode.
        _now = inference_config(policy, a.control_freq, a.max_episode_steps,
                                a.stock_init)
        for k, m in sorted(_tmeta.items()):
            diff = {f: (m[f], _now[f]) for f in MUST_MATCH
                    if m.get(f) is not None and m[f] != _now[f]}
            if diff:
                print(f"WARNING: {k} was searched under "
                      + ", ".join(f"{f}={was} (now {now})"
                                  for f, (was, now) in sorted(diff.items()))
                      + " -- that ticket was optimised for a different policy "
                        "than the one about to run.")
    if a.init_state_offset:
        print(f"[init] layouts offset by {a.init_state_offset} -- NOT the "
              f"canonical 0-19, so this run is not comparable to the tracker")
    cams = _policy_cameras(policy.config)
    print(f"[eval] {ckpt}  device={device}  cameras={cams}\n"
          f"[eval] horizon={policy.config.horizon} "
          f"n_action_steps={policy.config.n_action_steps} "
          f"NFE={policy.config.num_inference_steps} "
          f"state_history={policy.config.n_obs_steps} "
          f"noise={'GOLDEN TICKET (constant x_1)' if (a.noise_ticket or a.noise_tickets) else 'fixed/episode' if a.fixed_episode_noise else 'per-chunk'}")

    if a.state_noise > 0.0:
        # Report the physical size too. A sigma the arm cannot actually be off
        # by measures nothing, and one large enough to contradict the camera is
        # measuring a broken observation rather than a brittle policy.
        sc = state_scale(pre)
        dims = a.state_noise_dims if a.state_noise_dims is not None else "all"
        phys = ""
        if sc is not None:
            scale, kind = sc
            idx = (a.state_noise_dims if a.state_noise_dims is not None
                   else range(min(3, len(scale))))
            vals = [a.state_noise * float(scale[i]) for i in idx if i < len(scale)]
            if vals:
                phys = (f"  ~= {min(vals) * 1000:.1f}-{max(vals) * 1000:.1f} mm "
                        f"on dims {list(idx)} (from dataset {kind})")
        print(f"[eval] STATE NOISE sigma={a.state_noise} normalized on dims "
              f"{dims}{phys}\n"
              f"       One offset per window, so motion differences are "
              f"unchanged. This is a DIAGNOSTIC: flat SR means state "
              f"augmentation buys nothing.")

    if a.image_blur > 1:
        print(f"[eval] IMAGE BLUR x{a.image_blur} on "
              f"{a.image_blur_cams or 'every camera'} -- detail finer than "
              f"~{a.image_blur} px is gone, token grid unchanged.\n"
              f"       DIAGNOSTIC: flat SR means more vision resolution is not "
              f"the missing ingredient.")

    video_cb = make_video_writer(Path(a.video_dir) if a.video_dir else None,
                                 a.videos_per_task, a.videos_of)

    from lerobot.envs.libero import _get_suite

    results, t0 = {}, time.time()
    _ticketed = {}
    for suite_name in a.suites:
        suite = _get_suite(suite_name)
        n_tasks = getattr(suite, "n_tasks", None) or len(suite.tasks)
        task_ids = a.task_ids if a.task_ids is not None else list(range(n_tasks))
        # Read the instructions off the suite rather than off an env: LiberoEnv
        # binds one task at construction, so collecting them the other way
        # would mean building (and rendering) every task just to read a string.
        wrong = {}
        if a.ablate_lang:
            if n_tasks < 2:
                raise SystemExit(
                    f"--ablate_lang needs a suite with >1 task; {suite_name} "
                    f"has {n_tasks}, so the 'wrong' instruction would be the "
                    f"right one and the run would report a false null.")
            all_desc = {t: suite.get_task(t).language for t in range(n_tasks)}
            # A fixed half-suite offset: deterministic, and it never lands on
            # the task itself. Every libero_spatial task shares one tabletop,
            # so the wrong instruction is still valid FOR THAT SCENE -- it asks
            # for a different object, which is exactly the confusion to test.
            # A random or out-of-scene string would test novelty instead.
            wrong = {t: all_desc[(t + max(n_tasks // 2, 1)) % n_tasks]
                     for t in task_ids}
        elif a.instruction_override or a.instruction_from_task is not None:
            # --ablate_lang's half-suite offset asks "does ANY wrong instruction
            # change behaviour". That is the wrong question for a specific
            # confusion: libero_spatial task 7 ("on the stove") is scored at 60%
            # with its failures reaching for the cabinet, and task 9 ("on the
            # wooden cabinet") at 50% -- a pair the offset never puts against
            # each other. Naming the instruction directly is what tests whether
            # the policy can tell those two apart.
            if a.instruction_from_task is not None:
                if not 0 <= a.instruction_from_task < n_tasks:
                    raise SystemExit(
                        f"--instruction_from_task {a.instruction_from_task} is "
                        f"outside {suite_name}'s 0..{n_tasks - 1}")
                text = suite.get_task(a.instruction_from_task).language
            else:
                text = a.instruction_override
            for t in task_ids:
                real = suite.get_task(t).language
                if text.strip() == real.strip():
                    # Silently scoring a task against its own instruction would
                    # look like "language has no effect" when nothing was
                    # actually swapped.
                    raise SystemExit(
                        f"task {t}'s own instruction is {real!r}, which is what "
                        f"the override supplies. That is a null test, not a "
                        f"result -- pick a different task or string.")
            wrong = {t: text for t in task_ids}

        tag = ("  [LANGUAGE ABLATED]" if a.ablate_lang
               else "  [INSTRUCTION OVERRIDDEN]" if wrong else "")
        print(f"\n=== {suite_name}: {len(task_ids)} tasks x {a.episodes} episodes"
              f"{tag} ===")
        per_task = {}
        ticketed = []
        for k, tid in enumerate(task_ids):
            t_task = time.time()
            if a.noise_tickets:
                import ticket_bundle as tb
                _k = tb.key(suite_name, tid)
                _tk = _tickets.get(_k)
                _md = _tmeta.get(_k, {})
                _bb = _md.get("beats_baseline", True)
                if _tk is not None and _bb is False and not a.use_weak_tickets:
                    # beats_baseline false has TWO sources and they want
                    # opposite explanations. The search sets it when its
                    # winner did not clear the search baseline. A person sets
                    # it, through `ticket_bundle.py disable`, when the ticket
                    # DID clear that baseline and then lost on the reported
                    # layouts -- libero_10.1 searched 14/15 against 0.81 and
                    # cost 20 points at eval. Printing the search's wording
                    # there says the opposite of what happened, and quotes
                    # numbers that contradict it in the same line.
                    _why = _md.get("disabled_note")
                    if _why:
                        print(f"  [ticket] task {tid}: {_k} is DISABLED by "
                              f"hand -- falling back to Gaussian. It searched "
                              f"{_md.get('search_success')}, so this is not a "
                              f"search failure; see the eval that motivated "
                              f"it. --use_weak_tickets to force.")
                    else:
                        print(f"  [ticket] task {tid}: {_k} did NOT beat its "
                              f"search baseline ({_md.get('search_success')} "
                              f"vs {_md.get('baseline_rate')}) -- falling back "
                              f"to Gaussian. A ticket that only matches "
                              f"Gaussian is worse than none: it also removes "
                              f"the per-chunk re-draw. --use_weak_tickets to "
                              f"force.")
                    _tk = None
                elif _tk is not None and _bb is None:
                    print(f"  [ticket] task {tid}: {_k} beat its search "
                          f"baseline but not significantly (p="
                          f"{_md.get('p_vs_baseline')}) -- using it; this run "
                          f"IS the test.")
                _v = (None if _tk is None
                      else torch.from_numpy(_tk).float().to(device))
                _is_cyc = _tk is not None and _tk.ndim == 3
                policy.model._noise_ticket = None if _is_cyc else _v
                policy.model._noise_cycle = _v if _is_cyc else None
                policy.model._noise_cycle_k = 0
                if _tk is not None:
                    ticketed.append(tid)
                print(f"  [ticket] task {tid}: "
                      + (f"{tb.key(suite_name, tid)} "
                         f"(searched {_tmeta.get(tb.key(suite_name, tid), {}).get('search_success', '?')}"
                         f" on layouts {_tmeta.get(tb.key(suite_name, tid), {}).get('init_state_offset', '?')}+)"
                         if _tk is not None else "NONE in bundle -> Gaussian"))
            n_ok, n_ep, mean_steps, n_chunks, desc, ep_ok = eval_task(
                policy, pre, post, suite, suite_name, tid, a.episodes,
                a.num_envs, device, a.max_episode_steps, a.seed, cams,
                a.policy_seed, video_cb,
                a.videos_per_task, a.heartbeat, wrong.get(tid),
                a.state_noise, a.state_noise_dims,
                a.image_blur, a.image_blur_cams, a.history_mode,
                routing_acc, motion_acc, action_offset,
                init_state_offset=a.init_state_offset)
            sr = 100.0 * n_ok / max(n_ep, 1)
            per_task[tid] = {"success_rate": sr, "n_success": n_ok,
                             "n_episodes": n_ep, "mean_success_steps": mean_steps,
                             "policy_chunks": n_chunks, "task": desc,
                             # Ordered by episode index == canonical init state,
                             # so two runs of this line are paired. See eval_task.
                             "episode_success": ep_ok}
            done_n, total_n = k + 1, len(task_ids)
            eta = (time.time() - t0) / done_n * (total_n - done_n) / 60
            print(f"  task {tid:2d}  SR {sr:5.1f}%  ({n_ok}/{n_ep})  "
                  f"steps~{mean_steps:.0f}  [{done_n}/{total_n}, "
                  f"{(time.time() - t_task) / 60:.1f} min, ETA {eta:.0f} min]  "
                  f"{desc}", flush=True)
        rates = [v["success_rate"] for v in per_task.values()]
        _ticketed[suite_name] = ticketed
        if a.noise_tickets and len(ticketed) not in (0, len(task_ids)):
            print(f"  NOTE: {len(ticketed)}/{len(task_ids)} tasks had a ticket "
                  f"({ticketed}); the rest ran Gaussian. The suite average "
                  f"below mixes two policies and is NOT comparable to a "
                  f"uniform run -- read the per-task numbers.")
        results[suite_name] = {
            "per_task": per_task,
            "avg": float(np.mean(rates)),
            "min": float(np.min(rates)),
            "n_zero_tasks": int(sum(1 for r in rates if r == 0.0)),
        }
        s = results[suite_name]
        print(f"  {suite_name}: avg {s['avg']:.1f}%  MIN {s['min']:.1f}%  "
              f"tasks at zero: {s['n_zero_tasks']}")

    all_rates = [v["success_rate"] for s in results.values()
                 for v in s["per_task"].values()]
    avg, mn = float(np.mean(all_rates)), float(np.min(all_rates))
    zeros = [(s, t) for s, d in results.items()
             for t, v in d["per_task"].items() if v["success_rate"] == 0.0]

    print(f"\n{'=' * 62}")
    print(f"OVERALL  avg {avg:.1f}%   per-task MIN {mn:.1f}%   "
          f"tasks at zero: {len(zeros)}")
    gate = avg >= 93.0 and mn > 5.0
    print(f"stage-A gate (avg >= 93 AND min > 5): {'PASS' if gate else 'FAIL'}")
    if zeros:
        print("Tasks at 0% — stage-B RL cannot recover these; a binary reward "
              "has no gradient where every rollout fails:")
        for s, t in zeros:
            print(f"  {s} task {t}: {results[s]['per_task'][t]['task']}")
    print(f"{(time.time() - t0) / 60:.1f} min")

    if a.ablate_lang:
        print("\nThis was a LANGUAGE ABLATION -- the policy was given another "
              "task's instruction.\nCompare it against a non-ablated run with "
              "the same checkpoint, seed, episodes\nand cap. An unchanged "
              "success rate means the instruction is not driving the\npolicy, "
              "which no amount of further training changes.")

    # Everything a later run must match to be comparable. `seed` and the eval
    # commit were missing, and both bit: the seed only started reaching the
    # policy in 92ec163, so a JSON written before it recorded a draw that
    # cannot be reproduced -- and nothing in the file said which side it was
    # on. A baseline you cannot re-run is not a baseline.
    if routing_acc is not None:
        r = routing_acc.report()
        if r:
            E = len(r["mean_weights"])
            print("\n=== routing (per denoising step, pre-noise weights) ===")
            print(f"  chunks measured           : {r['chunks']}")
            if r.get("dense"):
                print(f"  (router_top_k=0, DENSE -- set stats use the top "
                      f"{r['set_k']} of {E} so they stay informative; the "
                      f"WEIGHTS below are the real measurement)")
            print(f"  top-{r['set_k']} set CHANGES within a chunk: {100 * r['switch_rate']:.1f}%"
                  "   <- the ODE integrates a field whose definition moves")
            print(f"  first step (t=1, coarse)  : " +
                  "  ".join(f"E{i}={100 * w:.1f}%" for i, w in enumerate(r["first_step_weights"])))
            print(f"  last  step (t->0, fine)   : " +
                  "  ".join(f"E{i}={100 * w:.1f}%" for i, w in enumerate(r["last_step_weights"])))
            print(f"  early->late L1 distance   : {r['early_late_L1']:.3f} "
                  f"(0 = the depth bands are NOT specialising coarse vs fine; "
                  f"max 2.0)")
            print(f"  early/late top-k overlap  : {r['early_late_set_jaccard']:.3f} "
                  f"(1.0 = the same k experts run the whole trajectory)")
            print(f"  uniform reference         : {100.0 / E:.1f}% per expert\n")

    if motion_acc is not None:
        m = motion_acc.report()
        if m:
            print("\n=== end-effector motion per policy chunk ===")
            print(f"  {'':<10}{'episodes':>9}{'median (mm)':>13}{'still %':>9}"
                  f"{'max streak':>12}{'mean streak':>13}{'chunks':>8}")
            for lab, k in (("success", "success"), ("FAILURE", "failure")):
                d = m.get(k)
                if d:
                    print(f"  {lab:<10}{d['episodes']:>9}{1000 * d['median_disp_m']:>13.1f}"
                          f"{100 * d['still_frac']:>8.1f}%{d['max_still_streak_chunks']:>12}"
                          f"{d['mean_still_streak_chunks']:>13.1f}{d['mean_chunks']:>8.1f}")
            so, fa = m.get("success"), m.get("failure")
            if so and fa:
                print(f"  still = under {1000 * m['still_threshold_m']:.0f} mm between chunk "
                      f"boundaries")
                # Compare ABSOLUTE motionless time, not the streak length.
                # Failures run far longer than successes (259 vs 80 chunks on
                # the first real measurement), so a streak ratio understates
                # them: 9.2 vs 3.4 reads as "only 2.7x" while the actual time
                # spent not moving is 8.4x. Report speed and time separately
                # rather than forcing a binary verdict -- the first real
                # measurement landed between the two hypotheses.
                still_s = so["mean_chunks"] * so["still_frac"]
                still_f = fa["mean_chunks"] * fa["still_frac"]
                spd = so["median_disp_m"] / max(fa["median_disp_m"], 1e-9)
                print(f"  speed:  successes move {spd:.1f}x further per chunk")
                print(f"  stalls: failures spend {still_f:.0f} chunks motionless vs "
                      f"{still_s:.0f} ({still_f / max(still_s, 1e-9):.1f}x)")
                if spd < 1.5 and still_f < 2 * still_s:
                    print("  -> failures look like successes: the arm is MISSING, not "
                          "stuck. A retreat would not help; this is precision.")
                elif spd > 2.5 and fa["still_frac"] > 0.5:
                    print("  -> failures are mostly frozen: the arm is WEDGED. Escape "
                          "(scripted retreat, adaptive noise, staged RL) is the lever.")
                else:
                    print("  -> DITHERING: failures still move on most chunks but at a "
                          "fraction of the speed, with long pauses. Neither pure "
                          "precision nor pure wedging -- watch the videos of the "
                          "slowest failures before choosing a lever.")
            print()

    payload = {"checkpoint": str(ckpt), "control_freq": a.control_freq,
               "fixed_init_states": not a.stock_init,
               "seed": a.seed,
               "policy_seed": a.policy_seed,
               # Not cosmetic: the policy draws one noise tensor of shape
               # (num_envs, ...) per chunk, so changing num_envs changes the
               # RNG stream and silently unpairs two runs.
               "num_envs": a.num_envs,
               "max_episode_steps": a.max_episode_steps,
               "eval_commit": _git_commit(),
               "num_inference_steps": getattr(policy.config, "num_inference_steps", None),
               "n_action_steps": policy.config.n_action_steps,
               "env": fingerprint(),
               "vision_input_size": getattr(policy.config, "vision_input_size", None),
               "temporal_ensemble_coeff": getattr(
                   policy.config, "temporal_ensemble_coeff", None),
               "stall_noise_scale": getattr(policy.config, "stall_noise_scale", None),
               "noise_ticket": a.noise_ticket,
               "noise_cycle": a.noise_cycle,
               "noise_cycle_m": (int(np.load(a.noise_cycle).shape[0])
                                 if a.noise_cycle else None),
               "noise_tickets": a.noise_tickets,
               # Which tasks ran with a ticket and which fell back. A suite
               # average over a PARTIAL bundle is two policies added together,
               # so it must never be reported without this line.
               "ticketed_tasks": (None if not a.noise_tickets else
                                  {sn: sorted(v) for sn, v in _ticketed.items()}),
               "init_state_offset": a.init_state_offset,
               "fixed_episode_noise": bool(a.fixed_episode_noise),
               "policy_type": getattr(policy.config, "type", None),
               "sample_noise_scale": getattr(
                   policy.config, "sample_noise_scale", None),
               "state_noise": a.state_noise,
               "history_mode": a.history_mode,
               "state_noise_dims": a.state_noise_dims,
               "image_blur": a.image_blur,
               "image_blur_cams": a.image_blur_cams,
               "router_top_k": getattr(policy.config, "router_top_k", None),
               "routing": (routing_acc.report() if routing_acc else None),
               "motion": (motion_acc.report() if motion_acc else None),
               "action_offset": ([[d, v] for d, v in action_offset]
                                 if action_offset else None),
               "episodes_per_task": a.episodes, "ablate_lang": a.ablate_lang,
               "instruction_override": a.instruction_override,
               "instruction_from_task": a.instruction_from_task,
               "overall_avg": avg, "overall_min": mn, "gate_pass": gate,
               "suites": results}
    # A separate filename: an ablation result overwriting the real one is a
    # mistake you only notice much later.
    default_name = ("eval_libero_ablated.json" if a.ablate_lang
                    else "eval_libero_override.json"
                    if (a.instruction_override or a.instruction_from_task is not None)
                    else f"eval_libero_statenoise_{a.state_noise:g}.json"
                    if a.state_noise > 0.0
                    else f"eval_libero_blur_{a.image_blur}.json"
                    if a.image_blur > 1
                    else f"eval_libero_temp_{a.sample_noise_scale:g}.json"
                    if a.sample_noise_scale is not None
                    else f"eval_libero_pseed_{a.policy_seed}.json"
                    if a.policy_seed is not None and a.policy_seed != a.seed
                    else "eval_libero.json")
    out = Path(a.out) if a.out else ckpt / default_name
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
