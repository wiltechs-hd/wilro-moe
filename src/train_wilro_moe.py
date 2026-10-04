import json
import time
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import huggingface_hub
from safetensors.torch import load_file as load_safetensors
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.datasets.compute_stats import aggregate_stats
import numpy as np
from torch.utils.data import ConcatDataset

# Wilro-specific components
from models.wilro_moe.wilro_moe_config import WilroMoEConfig
from models.wilro_moe.wilro_moe_policy import WilroMoEPolicy
from models.wilro_moe.processor_wilro_moe import make_pre_post_processors
from env_fingerprint import fingerprint_line
from task_rewrites import rewrite_instruction

from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup


# Detect the best available device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# Augmentation helpers (same recipe as train_transformer.py)
# ---------------------------------------------------------------------------
def get_augmentations():
    """Image augmentation transform shared across all cameras of a sample."""
    return v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])



class _TagDataset(torch.utils.data.Dataset):
    """Stamp which sub-dataset a sample came from.

    LeRobotDataset's `episode_index` is dataset-LOCAL and ConcatDataset does not
    renumber, so mixing the AWR corpus with the demo set would apply the
    corpus's weight for episode 5 to the demo set's episode 5 as well --
    silently, because the join is a dict lookup that cannot tell them apart.
    This is the disambiguator that makes mixing safe, and mixing is the right
    defence against the corpus being narrow: AWR trains on ONE dataset_id, so
    without it the model only ever sees the suite that was collected.
    """

    def __init__(self, ds, tag: int):
        self.ds, self.tag = ds, tag

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        sample = self.ds[i]
        sample["dataset_index"] = torch.tensor(self.tag, dtype=torch.long)
        return sample

    def __getattr__(self, name):          # meta, stats, fps, ... pass through
        # Guard the dunders. Unpickling builds the instance WITHOUT __init__, so
        # __dict__ is empty when pickle looks for __setstate__/__reduce_ex__;
        # routing that through here raises KeyError('ds') and the whole
        # DataLoader dies. It has not bitten yet only because Linux forks its
        # workers and fork does not pickle -- it would fail instantly under
        # spawn, i.e. on macOS/Windows or with multiprocessing_context="spawn".
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            ds = self.__dict__["ds"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(ds, name)


class AWRWeights:
    """Advantage-weighted regression weights, joined to episodes by index.

    The closed-form solution of "improve the policy but stay within KL eps of
    the data policy" is pi* = mu * exp(A/beta) / Z. Projecting that onto the
    parametric family is weighted maximum likelihood on the data you already
    have -- so the whole method is the existing flow-matching loss with a
    per-sample weight, and there is NO importance ratio: the weight comes from
    the reward and does not move as theta does.

    That is the property that matters here. GRPO's ratio is what exploded
    (485,165,216 = exp(20), the clamp) and what forces on-policy data. AWR has
    none, so a corpus collected once can be trained on for many epochs, and the
    gradient keeps the character of supervised learning rather than of a
    noise-dominated policy gradient -- which is what walked the policy off the
    SFT peak, 20.5% -> 5.4%.

    RFT is the beta -> 0 limit of this with a binary advantage: failures get
    weight 0 and are discarded. AWR keeps them at a low weight instead -- but
    be clear about what that low weight MEANS. There are no negative weights
    here: the loss is w * ||v - u||^2, so every w > 0 is an ATTRACTION. A
    failure at weight 0.26 does not say "not that way", it says "this way, but
    gently". The only repulsion AWR has is relative -- successes pull harder --
    and that is not enough when the failures share a systematic behaviour, as
    the never-grasped goal rollouts did. Saying "don't" needs a different
    algorithm: a policy gradient whose negative advantage carries a real minus
    sign, or a paired/contrastive objective. Hence --awr_drop_failures.
    """

    def __init__(self, path: str, beta: float, clip: float, kind: str,
                 dataset_index: int = 0, group_by: str = "task",
                 drop_failures: bool = False, min_weight: float = 0.0):
        # Which position in --dataset_id this sidecar describes. Every other
        # dataset in the mix gets weight 1.0, which is exactly right for demos:
        # they are the reference behaviour, not something to reweight.
        self.dataset_index = int(dataset_index)
        import json as _json
        d = _json.loads(Path(path).read_text())
        eps = d["episodes"]
        if not eps:
            raise ValueError(f"{path} lists no episodes")
        max_steps = max(int(e["steps"]) for e in eps) or 1
        if kind == "success":
            r = np.array([1.0 if e["success"] else 0.0 for e in eps], dtype=np.float64)
        elif kind == "fast_success":
            # A slow success fumbled and recovered; a fast one did not. On this
            # policy that is a real distinction -- successful episodes average
            # 80 chunks and failures 259 -- so it separates "it worked" from
            # "it worked eventually".
            r = np.array([(1.0 - 0.5 * e["steps"] / max_steps) if e["success"] else 0.0
                          for e in eps], dtype=np.float64)
        else:
            raise ValueError(f"unknown --awr_reward {kind!r}")

        # Standardise PER TASK by default: the reward scale is not comparable
        # across tasks, and without this beta means something different for
        # each one.
        #
        # group_by="state" standardises within (task, init_state) instead,
        # which is what GRPO's group baseline does and for the same reason --
        # it cancels LAYOUT DIFFICULTY so the advantage is about the policy.
        # Per task, `steps` conflates the two: a 200-step success may only mean
        # the object started far away. But it needs several rollouts PER STATE
        # to have anything to compare, and the collector's default sweep gives
        # exactly one, which makes every group degenerate (std=0 -> adv=0 ->
        # weight 1). At libero_10's 71% it takes k=5 rollouts per state to get
        # the degenerate share under 20%, i.e. 2500 episodes. Hence the default.
        tasks = np.array([e.get("task", "") for e in eps])
        if group_by == "state":
            if any("init_state" not in e for e in eps):
                raise ValueError(
                    "--awr_group_by state needs init_state in every sidecar "
                    "entry; this one predates the field. Re-collect, or use "
                    "--awr_group_by task.")
            keys = np.array([f"{e['task']}#{e['init_state']}" for e in eps])
        elif group_by == "task":
            keys = tasks
        else:
            raise ValueError(f"unknown --awr_group_by {group_by!r}")
        adv = np.zeros_like(r)
        n_degenerate = 0
        for t in np.unique(keys):
            m = keys == t
            sd = r[m].std()
            if sd <= 1e-6:
                n_degenerate += int(m.sum())
            adv[m] = (r[m] - r[m].mean()) / (sd if sd > 1e-6 else 1.0)
        w = np.clip(np.exp(adv / max(beta, 1e-6)), 0.0, clip)
        if min_weight > 0.0:
            # A floor on the low tail, applied BEFORE the zeroing below so
            # --awr_drop_failures still wins on failures. Use it when a small
            # beta has pushed the weaker successes to nearly nothing and the
            # corpus has effectively shrunk to its best few episodes.
            w = np.maximum(w, float(min_weight))
        n_dropped = 0
        if drop_failures:
            # The beta -> 0 (RFT) limit applied to the failure side only: keep
            # the graded weight among successes, discard the rest outright.
            #
            # The standardised weight does not suppress failures enough to be
            # harmless. On the goal corpus the gripper channel is bimodal +-1
            # in both the demos and the rollouts, and the only difference is
            # how often it is commanded closed: 47.5% of demo frames against
            # 31.2% of rollout frames. That gap IS the episodes that never
            # grasped. Carrying weight ~0.7 they still taught the policy to
            # hover, and libero_goal T0 -- "open the middle drawer of the
            # cabinet", a grasp-and-pull -- went 90% -> 35% after 4000 steps
            # of it.
            fail = r <= 0.0
            w = np.where(fail, 0.0, w)
            n_dropped = int(fail.sum())
            if not np.any(w > 0):
                raise ValueError(
                    f"{path}: --awr_drop_failures zeroed every episode. This "
                    f"corpus has no successes, so there is nothing left to "
                    f"weight -- collect more, or drop the flag and let the "
                    f"failures train at low weight.")
        # Renormalise to mean 1 so the loss scale -- and therefore the effective
        # learning rate -- does not move when beta or the success rate does.
        w = w / max(w.mean(), 1e-8)
        self.w = {int(e["episode_index"]): float(x) for e, x in zip(eps, w)}
        n_ok = int(sum(1 for e in eps if e["success"]))
        print(f"AWR weights: {len(eps)} episodes ({n_ok} success / {len(eps) - n_ok} "
              f"failure), reward={kind}, beta={beta:g}, clip={clip:g}, "
              f"group_by={group_by} ({len(np.unique(keys))} groups)")
        if n_degenerate:
            print(f"  [{'WARN' if n_degenerate > len(eps) // 2 else 'note'}] "
                  f"{n_degenerate}/{len(eps)} episodes sit in a group whose "
                  f"outcomes are all identical -- their advantage is 0 and they "
                  f"carry weight 1, i.e. they are plain SFT samples."
                  + ("  Over half the corpus: this grouping has too few "
                     "rollouts per group to say anything."
                     if n_degenerate > len(eps) // 2 else ""))
        print(f"  weight  min {w.min():.3f}  median {np.median(w):.3f}  "
              f"max {w.max():.3f}  (mean 1.000 by construction)")
        if n_dropped:
            surv = w[w > 0]
            print(f"  --awr_drop_failures: {n_dropped}/{len(eps)} episodes "
                  f"zeroed; the {len(surv)} survivors carry mean "
                  f"{surv.mean():.3f}. The corpus pulls on the run exactly as "
                  f"hard as before -- same total weight, now concentrated on "
                  f"the successes.")
        if n_ok:
            wo = w[[i for i, e in enumerate(eps) if e["success"]]]
            wf = w[[i for i, e in enumerate(eps) if not e["success"]]]
            print(f"  success mean {wo.mean():.3f}   failure mean "
                  f"{(wf.mean() if len(wf) else float('nan')):.3f}   "
                  f"ratio {(wo.mean() / wf.mean() if len(wf) and wf.mean() > 0 else float('inf')):.1f}x")

    def lookup(self, episode_index, dataset_index=None) -> "torch.Tensor":
        """Weight per sample; 1.0 for anything this sidecar does not describe.

        With several datasets mixed, `dataset_index` is what keeps the corpus's
        weights off the demo set. Passing None is only safe for a single
        dataset, and train() refuses the multi-dataset case without it.
        """
        if dataset_index is None:
            return torch.tensor([self.w.get(int(i), 1.0) for i in episode_index],
                                dtype=torch.float32)
        return torch.tensor(
            [self.w.get(int(e), 1.0) if int(d) == self.dataset_index else 1.0
             for e, d in zip(episode_index, dataset_index)],
            dtype=torch.float32)


class AWRWeightSet:
    """Several AWR corpora at once, each bound to one --dataset_id position.

    The collector does ONE suite per run, so covering libero_10 + goal +
    spatial + object means four sidecars and four collected datasets. Each one
    is standardised per task inside itself and renormalised to mean 1, so the
    corpora stay comparable to each other and to the unweighted demo set, and
    the mix ratio is set by their frame counts rather than by an accident of
    reward scale.

    Everything not named by a sidecar gets 1.0 -- the demo set included, which
    is the point: it is the reference behaviour holding the policy in place
    while the corpora pull on it.
    """

    def __init__(self, paths, indices, beta: float, clip: float, kind: str,
                 group_by: str = "task", drop_failures: bool = False,
                 min_weight: float = 0.0):
        paths = [p for p in paths if p]
        if len(indices) != len(paths):
            raise ValueError(
                f"--awr_rewards has {len(paths)} path(s) but "
                f"--awr_dataset_index has {len(indices)}; they are positional "
                f"pairs and must match.")
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"--awr_dataset_index {indices} repeats a position; two "
                f"sidecars cannot describe the same dataset.")
        self.by_ds = {int(i): AWRWeights(p, beta, clip, kind,
                                        dataset_index=int(i), group_by=group_by,
                                        drop_failures=drop_failures,
                                        min_weight=min_weight)
                      for p, i in zip(paths, indices)}
        self.indices = sorted(self.by_ds)

    def lookup(self, episode_index, dataset_index=None) -> "torch.Tensor":
        if dataset_index is None:
            if len(self.by_ds) != 1:
                raise KeyError(
                    "several AWR corpora but no dataset_index in the batch; "
                    "the weights cannot be told apart.")
            return next(iter(self.by_ds.values())).lookup(episode_index)
        out = []
        for e, d in zip(episode_index, dataset_index):
            w = self.by_ds.get(int(d))
            out.append(w.w.get(int(e), 1.0) if w is not None else 1.0)
        return torch.tensor(out, dtype=torch.float32)


def apply_joint_augmentations(batch, abs_sigma: float = 0.01,
                              frac_sigma: float = 0.0, state_std=None,
                              prob: float = 0.5):
    """Gaussian noise on observation.state, applied with probability `prob`.

    Runs BEFORE the preprocessor, so it is in RAW units -- metres for the
    end-effector position and the gripper finger joints, radians for the
    orientation. That matters, because `abs_sigma` is then one number spread
    over dims whose natural scales differ by 65x. On LIBERO the default 0.01
    lands as:

        eef x/y/z   9.5% / 6.6% / 2.6%  of that dim's own std
        rot 0/1/2   2.9% / 1.1% / 3.1%
        gripper L/R      70.5% / 71.1%   <-- 70% of the signal's own spread

    i.e. it is heaviest exactly on the channel that carries "am I holding it",
    and invisible on rotation. Nobody would choose that ratio; it is what one
    absolute sigma does to a heterogeneous state vector.

    `frac_sigma` scales per dim instead, so the knob means ONE thing everywhere:
    frac 0.04 is 4% of each dim's own std. It is off by default -- every result
    in notes/libero_benchmark_tracker.md was trained with the absolute path, and
    switching is a change, not a bug fix.

    NOTE the direction this teaches. The action is an OSC DELTA, not an absolute
    target, so perturbing the state while keeping the demo action label trains
    "produce the same motion regardless", i.e. INVARIANCE. The offset is carried
    forward, not corrected. Recovery would need the label compensated too
    (s+eps paired with d-eps); this does not do that.
    """
    if prob <= 0.0 or torch.rand(1).item() >= prob:
        return batch
    if "observation.state" not in batch:
        return batch
    st = batch["observation.state"]
    if frac_sigma > 0.0 and state_std is not None:
        sigma = frac_sigma * state_std.to(device=st.device, dtype=st.dtype)
        batch["observation.state"] = st + torch.randn_like(st) * sigma
    elif abs_sigma > 0.0:
        batch["observation.state"] = st + torch.randn_like(st) * abs_sigma
    return batch


def apply_image_augmentations(batch, camera_keys, transform):
    """Apply the same random color jitter to all cameras within each sample.

    For each sample in the batch, all camera images are stacked into a single
    tensor and passed through the transform in one call. torchvision v2 samples
    random parameters once per forward() call and applies them identically to
    every image in the tensor — so front/gripper/right cameras always receive
    the same brightness/contrast/saturation/hue shift, keeping cross-camera
    color consistency.

    Handles both (C, H, W) and (T, C, H, W) camera tensors.
    """
    present_keys = [k for k in camera_keys if k in batch and isinstance(batch[k], torch.Tensor)]
    if not present_keys:
        return batch

    B = batch[present_keys[0]].shape[0]
    for b in range(B):
        sample_img = batch[present_keys[0]][b]
        has_time_dim = sample_img.dim() == 4
        if has_time_dim:
            T = sample_img.shape[0]
            stacked = torch.cat([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i * T:(i + 1) * T]
        else:
            stacked = torch.stack([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i]
    return batch


# ---------------------------------------------------------------------------
# Gradient analysis tailored to wilro components
# ---------------------------------------------------------------------------
def _rss_gb():
    """Resident set of this process AND its dataloader workers, in GB.

    Workers are separate processes, so the parent's own RSS misses most of
    the growth that ends these runs."""
    try:
        import os
        def rss(pid):
            with open(f"/proc/{pid}/status") as f:
                for ln in f:
                    if ln.startswith("VmRSS:"):
                        return int(ln.split()[1]) / 1048576.0
            return 0.0
        me = os.getpid()
        try:
            with open(f"/proc/{me}/task/{me}/children") as f:
                kids = [int(x) for x in f.read().split()]
        except Exception:
            kids = []
        return rss(me) + sum(rss(k) for k in kids), len(kids)
    except Exception:
        return None, 0


_RSS_PREV: list = []


def _log_gradient_analysis(policy, step: int) -> None:
    print(f"\n--- Gradient Analysis at Step {step} ---")
    tot, nk = _rss_gb()
    if tot is not None:
        # The DELTA is the diagnostic, not the level. A large constant baseline
        # is just the model, the CUDA context and the Arrow table; what kills
        # the run is growth, and the OOM killer takes a WORKER, which surfaces
        # as "DataLoader worker (pid ...) is killed by signal: Killed" raised
        # from wherever the main process happened to be -- never from the loader.
        d = f"{tot - _RSS_PREV[-1]:+.2f} GB since step {_RSS_PREV[0]:.0f}" \
            if _RSS_PREV else "baseline"
        note = ""
        if _RSS_PREV and tot - _RSS_PREV[-1] > 0.3:
            note = ("   <-- GROWING; the workers are un-sharing the dataset. "
                    "Lower --num_workers / --prefetch_factor before it is "
                    "killed.")
        print(f"  Host RAM (self + {nk} worker(s)): {tot:.1f} GB  ({d}){note}")
        _RSS_PREV[:] = [step, tot]

    def _grad_stats(prefix: str):
        total, count = 0.0, 0
        for name, param in policy.model.named_parameters():
            if param.requires_grad and prefix in name and param.grad is not None:
                total += param.grad.abs().mean().item() * param.numel()
                count += param.numel()
        return (total / count, count) if count > 0 else (None, 0)

    for label, prefix in [
        ("Vision LoRA",      "vision_model.encoder.layers"),  # SigLIP ViT LoRA (trainable)
        ("Text LoRA",        "text_model.layers"),            # Text model LoRA (trainable)
        ("Connector (frzn)", "connector"),
        ("State Enc",        "state_encoder"),
        ("ResNet",           "robot_visual_encoder"),
        ("Experts",          "experts"),
        ("  ├─ Self-attn",   "sa_"),
        ("  ├─ VLM KV CA",   "ca_"),
        ("  └─ FFN",         "ffn"),
        ("Router",           "router"),
        ("Expert vis adapt", "expert_vision_adapter"),
        ("Action In/Out",    "action_"),
        ("Sink token",       "sink_token"),
        ("Final Norm",       "final_norm"),
        ("Time MLP",         "time_embedder"),
    ]:
        grad, n = _grad_stats(prefix)
        if grad is not None:
            print(f"  {label:22s} - Avg Abs Grad: {grad:.3e} ({n:,} params)")

    stats = getattr(policy.model, "_last_attention_stats", None)
    if stats:
        # Match expert sequence order: [SINK, state, vision, action]
        order = ["sink", "state", "vision", "action"]
        ordered = [(k, stats[k]) for k in order if k in stats]
        cells = "  ".join(f"{k}={v*100:5.1f}%" for k, v in ordered)
        print(f"  Action→ self-attn : {cells}    (last DiT layer)")

    x_stats = getattr(policy.model, "_last_cross_attention_stats", None)
    if x_stats:
        # VLM cross-attention: vision vs language
        vlm_order = ["vision", "language"]
        vlm_ordered = [(k, x_stats[k]) for k in vlm_order if k in x_stats]
        vlm_cells = "  ".join(f"{k}={v*100:5.1f}%" for k, v in vlm_ordered)
        print(f"  Action→ VLM x-attn  : {vlm_cells}    (cross-attn to VLM KV)")

    # Robot cross-attention stats (if captured)
    robot_ca_stats = getattr(policy.model, "_last_robot_cross_attention_stats", None)
    if robot_ca_stats:
        robot_cells = "  ".join(f"{k}={v*100:5.1f}%" for k, v in robot_ca_stats.items())
        print(f"  Action→ Robot x-attn: {robot_cells}    (cross-attn to Robot CNN)")

    usage = getattr(policy.model, "_last_router_usage", None)
    if usage is not None:
        u = usage.detach().float().cpu()
        cells = "  ".join(f"E{i}={v*100:5.1f}%" for i, v in enumerate(u.tolist()))
        cv2 = float((u.std(unbiased=False) / u.mean().clamp(min=1e-8)).pow(2))
        print(f"  Router usage      : {cells}    CV^2={cv2:.4f}")
        # usage is a batch MEAN, so a flat CV^2 is ambiguous -- every sample can
        # be fully collapsed and still average out uniform if different samples
        # collapse to different experts. These two read the PRE-noise per-sample
        # weights, which is what inference uses.
        # AMBIGUITY is the one to read. By Krogh-Vedelsby it is exactly what
        # the mixture subtracts from the mean individual MSE, so as a fraction
        # of the flow loss it IS the ensemble gain -- no threshold needed. The
        # dimensionless `disagreement` is kept for continuity but its upper
        # anchor moves with num_experts (0.997 at 2, 1.155 at 4, 1.193 at 6),
        # so it does not compare across configurations.
        amb = getattr(policy.model, "_last_expert_ambiguity", None)
        dis = getattr(policy.model, "_last_expert_disagreement", None)
        if amb is not None or dis is not None:
            _c = getattr(policy.model, "_last_loss_components", None) or {}
            flow = _c.get("main")
            # The verdict below is MEANINGLESS until the adaLN-Zero gates have
            # opened. At init every expert is the identity map and
            # action_out_proj is zero, so v_e == 0 for every e, ambiguity is
            # exactly 0, and the flow loss still sits at its "predict nothing"
            # value -- the ratio would read 0.0% and scream "redundant" on every
            # run's first thousand steps. Warmup is the honest boundary: before
            # 2x warmup the LR has not been at peak for any meaningful span.
            warm = int(getattr(policy.model.config, "scheduler_warmup_steps", 1500) or 1500)
            # Gate on DIFFERENTIATION, not on the step counter alone. The step
            # guard exists because adaLN-Zero makes every expert the identity
            # map at init, so the statistic reads ~0 whatever the topology is
            # worth. But --start_step_override 0 restarts the counter on
            # ALREADY-TRAINED weights: step 0 with disagreement 0.574 and
            # ambiguity at 74% of flow, where the "too early to read" message
            # is not merely unhelpful, it is false. Disagreement is the direct
            # test of the thing the step counter was standing in for.
            settled = step >= 2 * warm or (dis is not None and dis > 0.05)
            if amb is not None and flow and flow > 0:
                pct = 100.0 * amb / flow
                # The two are normalised slightly differently -- the flow loss
                # carries the position/dim weights and this does not -- so read
                # the percentage to within ~20%, and read its TREND exactly.
                if not settled:
                    verdict = (f"  <- too early to read (step {step} < 2x warmup "
                               f"{warm} and the experts have not differentiated); "
                               f"adaLN-Zero starts every expert as the identity "
                               f"map, so this reads ~0 whatever the topology is "
                               f"worth")
                elif pct < 1.0:
                    verdict = ("  <- the experts are near-redundant; these "
                               "parameters would do more as depth "
                               "(--num_experts 1 --expert_num_layers 32, "
                               "same params, same FLOPs)")
                elif pct < 5.0:
                    verdict = "  <- small but real ensemble gain"
                else:
                    verdict = ("  <- the mixture is doing real work; check the "
                               "flow loss too, since 'all bad in different "
                               "ways' looks the same here")
                print(f"                      expert ambiguity={amb:.5f} "
                      f"= {pct:.1f}% of flow {flow:.4f}{verdict}")
            elif amb is not None:
                print(f"                      expert ambiguity={amb:.5f} "
                      f"(no flow loss recorded to compare against)")
            if dis is not None:
                E = int(u.numel())
                iid = {2: 0.997, 4: 1.155, 6: 1.193, 8: 1.209}.get(E)
                anchor = f" (independent-expert anchor {iid:.3f})" if iid else ""
                hint = ("  <- adaLN-Zero: experts are still the identity map, "
                        "so this is 'not differentiated yet', NOT 'in "
                        "agreement'" if dis < 1e-4 else "")
                print(f"                      expert disagreement={dis:.4f}"
                      f"{anchor}{hint}")
        mw = getattr(policy.model, "_last_router_max_w", None)
        ent = getattr(policy.model, "_last_router_entropy", None)
        if mw is not None and ent is not None:
            E = int(u.numel())
            import math as _m
            print(f"                      per-sample max_w={mw:.3f} "
                  f"(uniform {1.0 / E:.3f})   entropy={ent:.3f} "
                  f"(uniform {_m.log(E):.3f})")

    # ---- zero-init gates ----------------------------------------------------
    # Both of these start at EXACTLY 0 so that a pathway which never earns its
    # keep is the identity map rather than noise. The cost of that safety is
    # that a pathway which never opens is INVISIBLE: the run completes, the
    # tokens are computed and multiplied by ~0, and nothing in the loss curve
    # says so. resnet_motion_tokens in particular doubles per-camera video
    # decode to build its second frame, so a gate stuck at 0 means paying that
    # for nothing.
    _gates = [("resnet_motion_gate", getattr(policy.model, "resnet_motion_gate", None)),
              ("expert_vision_gates", getattr(policy.model, "expert_vision_gates", None))]
    _gates = [(n, g) for n, g in _gates if g is not None]
    if _gates:
        warm = int(getattr(policy.model.config, "scheduler_warmup_steps", 1500) or 1500)
        print("  Zero-init gates    :")
        for name, g in _gates:
            v = g.detach().float().flatten()
            gr = (g.grad.detach().float().flatten().abs().mean().item()
                  if g.grad is not None else float("nan"))
            vals = "  ".join(f"{x:+.3e}" for x in v.tolist())
            print(f"    {name:<22}{vals}   |grad| {gr:.3e}")
            peak = float(v.abs().max())
            if step >= 2 * warm and peak < 1e-3:
                print(f"      [WARN] still {peak:.1e} after {step} steps -- this "
                      f"pathway is effectively OFF. Its tokens are being computed "
                      f"and multiplied by ~0"
                      + (", and the second camera frame that feeds it is doubling "
                         "video decode for nothing" if name == "resnet_motion_gate"
                         else "") + ".")
            elif step < 2 * warm:
                print(f"      (too early to judge: zero-init, step {step} "
                      f"< 2x warmup {warm})")

    comps = getattr(policy.model, "_last_loss_components", None)
    cw = getattr(policy.model.config, "contrastive_loss_weight", 0.0)
    if comps is not None and cw > 0.0:
        margin = getattr(policy.model.config, "contrastive_margin", 0.05)
        main_v = comps.get("main", float("nan"))
        contr_v = comps.get("contrastive", float("nan"))
        pct = (contr_v / margin * 100.0) if margin > 0 else float("nan")
        print(f"  Contrastive       - main: {main_v:.4f}   contrastive: {contr_v:.4f} "
              f"({pct:.0f}% of margin {margin:.3f})   weight: {cw}")
        if "diff_sq_mean" in comps:
            # How far the permuted-language prediction actually sits from the
            # correct one. A hinge of 0.0000 is compatible with a separation of
            # 0.051 and with one of 5.0; only this tells them apart, and it is
            # what says whether the margin is set anywhere near the action.
            print(f"                      separation diff_sq  mean "
                  f"{comps['diff_sq_mean']:.4f}  p10 {comps['diff_sq_p10']:.4f}  "
                  f"min {comps['diff_sq_min']:.4f}   "
                  f"{comps['diff_sq_under'] * 100:.0f}% of "
                  f"{int(comps['n_pairs'])} pairs under the margin   "
                  f"({'HARD' if comps.get('hard_neg') else 'random'} negatives)")
            # Against the main loss, not against the margin. The margin is an
            # arbitrary constant; the flow loss is the model's own error scale,
            # so separation/main says whether permuting the instruction moves
            # the prediction by more or less than the model is already wrong by.
            if main_v == main_v and main_v > 0:
                rel = comps["diff_sq_mean"] / main_v
                verdict = ("language is LOAD-BEARING: permuting it moves the "
                           "prediction further than the model's own error, so "
                           "the hinge reading 0.0000 means satisfied, not broken"
                           if rel >= 1.0 else
                           "permuting the instruction moves the prediction LESS "
                           "than the model's own error -- the pressure is worth "
                           "applying")
                print(f"                      separation is {rel * 100:.0f}% of "
                      f"the main loss {main_v:.4f} -- {verdict}")
            if comps["diff_sq_under"] < 0.05 and margin > 0:
                print(f"                      [note] only "
                      f"{comps['diff_sq_under'] * 100:.0f}% of pairs are under "
                      f"the margin, so this term is numerically INERT. Making "
                      f"it bite needs --contrastive_margin near the MEAN "
                      f"separation ({comps['diff_sq_mean']:.2f}), not near p10 "
                      f"-- but read the ratio above first: if separation "
                      f"already exceeds the main loss, raising the margin "
                      f"demands more language sensitivity than any measurement "
                      f"says is missing.")

    print("--- End Gradient Analysis ---\n")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(output_dir, dataset_id="lerobot/libero", resume_from_checkpoint=None,
          gradient_checkpointing=False, max_episode_index=None, batch_size=64,
          contrastive_loss_weight=0.1, contrastive_margin=0.05,
          contrastive_hard_negatives=False,
          lock_joint_index: int | None = None, kv_capture_strategy: str = "last",
          kv_capture_layers: list | None = None,
          cameras: list | None = None,
          rewrite_instructions: bool = False,
          rewrite_augment: bool = False,
          noise_temporal_correlation: float = 0.0,
          gripper_phase_weight: float = 1.0,
          video_backend: str | None = None,
          n_action_steps_cli: int | None = None,
          gripper_transition_window: int = 2,
          gripper_transition_thresh: float = 0.5,
          time_sampling: str = "uniform",
          time_lognormal_mean: float = -0.5,
          time_lognormal_std: float = 1.0,
          paraphrase_augment: bool = False,
          paraphrase_limit: int = 8,
          paraphrase_file: str = "",
          paraphrase_min_variants: int = 5,
          training_steps: int | None = None,
          n_obs_steps: int | None = None,
          val_episodes: int = 0,
          val_every: int = 500,
          val_max_batches: int = 20,
          progress_update_freq: int = 200,
          num_workers: int = 8,
          start_step_override: int = -1,
          lr: float | None = None,
          warmup_steps: int | None = None,
          lora_rank: int | None = None,
          lora_alpha: float | None = None,
          vision_lora_num_layers: int | None = None,
          download_progress: bool = False,
          cache_sync: bool = False,
          load_image_size: int = 0,
          prefetch_factor: int = 2,
          vision_token_source: str = "vlm",
          resnet_tokens: int = 64,
          awr_rewards=(),
          awr_dataset_index=(0,),
          awr_group_by: str = "task",
          awr_beta: float = 1.0,
          awr_clip: float = 20.0,
          awr_reward: str = "success",
          awr_drop_failures: bool = False,
          awr_min_weight: float = 0.0,
          state_noise_abs: float = 0.01,
          state_noise_frac: float = 0.0,
          state_noise_prob: float = 0.5,
          resnet_fine_cameras: list | None = None,
          resnet_fine_tokens: int = 0,
          resnet_input_size: int = 256,
          vision_input_size: int = 384,
          resnet_pool: str = "avg",
          use_state_history: bool = False,
          resnet_motion_tokens: int = 0,
          resnet_motion_stride: int = 1,
          num_experts: int = 4,
          expert_num_layers: int = 8,
          dit_hidden_size: int = 960,
          vlm_capture_layers: str = "",
          resnet_expert_adapter_dim: int = 0,
          router_temperature: float = 1.0,
          router_top_k: int = 0,
          router_balance_weight: float = 0.1):
    """Train the Wilro (SmolVLM2 KV-cache → DiT) flow matching model.

    `dataset_id` may be a single id or a list. Multiple datasets are concatenated
    and assumed HOMOGENEOUS (same robot / cameras / state+action dims / fps) — e.g.
    several piper sets — and their normalization stats are aggregated. For
    mixed-robot data use the canonical train_finetune.py path instead.
    """
    dataset_ids = [dataset_id] if isinstance(dataset_id, str) else list(dataset_id)
    if not dataset_ids:
        raise ValueError("At least one dataset_id is required.")

    # Argument-only preflight, before a single byte is downloaded. Each of these
    # combinations is accepted downstream and produces a run that completes and
    # answers a different question than the one asked.
    # SmolVLM2 has 32 text layers where Qwen3-VL-4B has 36, and the experts'
    # KV bands are disjoint, so wiltechs_moe's 4 x 9 = 36 does not port. Caught
    # here rather than after the VLM download, which is where the model's own
    # check fires.
    _need = int(num_experts) * int(expert_num_layers)
    if not vlm_capture_layers and _need > 32:
        raise ValueError(
            f"--num_experts {num_experts} x --expert_num_layers {expert_num_layers} "
            f"= {_need} exceeds SmolVLM2-500M's 32 text layers. 4 x 8 = 32 fits "
            f"exactly and is the default.")
    if _need % int(num_experts) != 0:
        raise ValueError("capture-layer count must divide by --num_experts")
    if int(dit_hidden_size) != 960:
        raise ValueError(
            f"--dit_hidden_size {dit_hidden_size}: only 960 (the VLM's hidden "
            f"size) is supported. Below it the expert self- and cross-attention "
            f"need different head geometries, and this model reuses wilro's "
            f"single-geometry DiTLayer.")
    if vision_token_source not in ("vlm", "resnet"):
        raise ValueError(f"--vision_token_source must be vlm or resnet, "
                         f"got {vision_token_source!r}")
    if resnet_motion_tokens > 0 and vision_token_source != "resnet":
        raise ValueError(
            "--resnet_motion_tokens needs --vision_token_source resnet. The motion "
            "tokens are produced by the ResNet's own feature maps; under the "
            "vlm source there is no encoder to difference and the "
            "extra camera frame would be decoded and thrown away.")
    if vision_token_source == "resnet" and resnet_pool == "avg":
        _side = int(resnet_tokens ** 0.5)
        if _side * _side != resnet_tokens:
            raise ValueError(f"--resnet_tokens must be a perfect square for "
                             f"avg pooling, got {resnet_tokens}")
    if use_state_history and (n_obs_steps is None or int(n_obs_steps) < 2):
        raise ValueError(
            f"--use_state_history with --n_obs_steps {n_obs_steps} enables nothing: "
            f"the window would be one frame, so removing the slice still leaves one "
            f"state token. Pass --n_obs_steps 8 (the width the leak control in "
            f"notes/wiltechs_x_ablations.md was run at).")

    # huggingface_hub draws a per-file tqdm for every repo it touches, and this
    # trainer touches each dataset twice (metadata, then the dataset itself).
    # At ~14k files that is thousands of redrawn lines before the first step,
    # which buries the schema and normalisation output that actually needs
    # reading. One status line each instead; --download_progress restores the
    # bars for a genuinely first-time pull.
    if not download_progress:
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except Exception:
            pass
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # The cosine schedule spans this, so it is not a "stop whenever" ceiling:
    # interrupting a 200k run at 30k leaves the LR mid-cosine and the model
    # never annealed. Pick the number you intend to finish.
    steps_cli = training_steps                       # None unless asked for
    training_steps = 200000 if steps_cli is None else int(steps_cli)
    progress_update_freq = max(1, int(progress_update_freq))
    checkpoint_freq = 1000
    image_transforms = get_augmentations()

    # Load metadata for all datasets. Schema is taken from the first and the rest
    # are validated against it (homogeneous assumption).
    # force_cache_sync re-verifies every file in the repo against the hub on
    # each launch -- ~14k HEAD requests for a converted VLABench, before a
    # single step. It only matters when the remote may have changed under a
    # cache that already exists, so it is opt-in via --cache_sync.
    metas = {}
    for did in dataset_ids:
        _t = time.time()
        print(f"[data] reading metadata for {did}"
              + ("  (--cache_sync: re-verifying every file against the hub)"
                 if cache_sync else ""), flush=True)
        metas[did] = LeRobotDatasetMetadata(did, force_cache_sync=cache_sync,
                                            revision="main")
        print(f"[data]   ...{time.time() - _t:.0f}s", flush=True)
    ref_meta = metas[dataset_ids[0]]
    features = dataset_to_policy_features(ref_meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    if len(output_features) == 0:
        raise ValueError("No output features (actions) found! Check your dataset schema.")

    print('input_features:', input_features)
    print('output_features:', output_features)

    # Detect all available cameras from dataset features
    all_camera_keys = sorted([key for key, ft in input_features.items() if ft.type is FeatureType.VISUAL])
    
    # Filter cameras if --cameras is specified
    if cameras is not None and len(cameras) > 0:
        camera_keys = [c for c in cameras if c in all_camera_keys]
        missing = [c for c in cameras if c not in all_camera_keys]
        if missing:
            print(f"WARNING: Requested cameras not found in dataset: {missing}")
        if not camera_keys:
            raise ValueError(
                f"None of the requested cameras {cameras} exist in dataset. "
                f"Available cameras: {all_camera_keys}"
            )
        print(f"Camera filter applied: using {camera_keys} (from available {all_camera_keys})")
    else:
        camera_keys = all_camera_keys
    
    state_dim = input_features["observation.state"].shape[-1] if "observation.state" in input_features else 7
    action_dim = next(iter(output_features.values())).shape[-1]
    print(f"Detected cameras ({len(camera_keys)}): {camera_keys}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Validate the other datasets share the same schema.
    # Image RESOLUTION is tracked separately from the schema check below. It is
    # not a schema conflict -- the model pads to square and interpolates every
    # frame to vision_input_size regardless -- but ConcatDataset collates raw
    # tensors, so two sets at 480 and 256 pass every check here and then die in
    # torch.stack, inside a worker if num_workers > 0:
    #   "stack expects each tensor to be equal size, but got [3, 256, 256] at
    #    entry 0 and [3, 480, 480] at entry 3"
    # Resolved by resizing at LOAD time instead (see resize_to below), which is
    # the same operation the model would do later and therefore lossless.
    vis_shapes = {dataset_ids[0]: {k: tuple(ft.shape)
                                   for k, ft in input_features.items()
                                   if ft.type is FeatureType.VISUAL}}
    for did in dataset_ids[1:]:
        f = dataset_to_policy_features(metas[did].features)
        out_f = {k: ft for k, ft in f.items() if ft.type is FeatureType.ACTION}
        in_f = {k: ft for k, ft in f.items() if k not in out_f}
        cks = sorted(k for k, ft in in_f.items() if ft.type is FeatureType.VISUAL)
        sd = in_f["observation.state"].shape[-1] if "observation.state" in in_f else 7
        ad = next(iter(out_f.values())).shape[-1]
        vis_shapes[did] = {k: tuple(ft.shape) for k, ft in in_f.items()
                           if ft.type is FeatureType.VISUAL}
        if cks != camera_keys or sd != state_dim or ad != action_dim:
            raise ValueError(
                f"Dataset '{did}' schema differs from '{dataset_ids[0]}':\n"
                f"  cameras {cks} vs {camera_keys}\n"
                f"  state_dim {sd} vs {state_dim}, action_dim {ad} vs {action_dim}\n"
                f"train_wilro_moe.py concatenation requires a homogeneous schema. For "
                f"mixed robots use the canonical train_finetune.py path."
            )

    # Aggregate normalization stats across datasets (count-weighted mean, global
    # min/max, combined std). One dataset keeps its own stats unchanged.
    if len(dataset_ids) == 1:
        combined_stats = ref_meta.stats
    else:
        combined_stats = aggregate_stats([metas[did].stats for did in dataset_ids])
        print(f"Aggregated normalization stats across {len(dataset_ids)} datasets.")

    # Training parameters — match train_transformer.py for like-for-like comparison
    obs = 2 if n_obs_steps is None else max(1, int(n_obs_steps))
    horizon = 64
    # TRAINING-SIDE ONLY. It is the boundary in compute_loss:
    #   pos_w[n_action_steps:] = future_steps_weight
    # so with the historical 64 that slice is EMPTY and future_steps_weight has
    # never once been in effect -- all 64 positions trained at weight 1.0 while
    # eval executes 2. The 2026-09-13 horizon profile measured the executed
    # prefix as the WORST bucket of the chunk (0.213 of "predict nothing"
    # against 0.153 at positions 16-31) while it carried 3.1% of the gradient.
    # 8 puts that at 8.1%. Inference is unaffected: every eval overrides
    # n_action_steps on the command line.
    n_action_steps = 64 if n_action_steps_cli is None else int(n_action_steps_cli)
    if n_action_steps > horizon:
        raise SystemExit(
            f"--n_action_steps {n_action_steps} exceeds horizon {horizon}; "
            f"there are no chunk positions past the horizon to weight.")
    if n_action_steps == horizon:
        # Not an error -- it is the historical default and every checkpoint in
        # this repo carries it -- but it must not pass silently, because it is
        # the reason future_steps_weight has never done anything.
        print(f"  NOTE n_action_steps == horizon ({horizon}): pos_w"
              f"[{n_action_steps}:] is an EMPTY slice, so future_steps_weight is "
              f"INERT and all {horizon} positions train at weight 1.0. The "
              f"executed prefix then carries {2 / horizon * 100:.1f}% of the "
              f"gradient. Pass --n_action_steps 8 to give it a real boundary.")

    # Build action_dim_weights FROM THE DATA, not from a flag default.
    #
    # A zero weight does not suppress a dim, it randomises it. action_out_proj
    # is zero-init; with weight 0 that row gets no gradient from the flow loss,
    # and none from the contrastive term either (v_t and v_wrong share the row,
    # so their difference on that dim is identically 0). The row stays at zero
    # forever => v_t[..., i] == 0 => the Euler loop never moves x_t[..., i] =>
    # the emitted value is the INITIAL NOISE, unnormalized: a fresh draw from
    # that dim's marginal on every single step.
    #
    # That is harmless only when the dim is genuinely constant, where the
    # marginal is a point mass. When it is not, sampling the marginal costs
    # 2*sigma^2 against sigma^2 for simply emitting the mean -- i.e. a zero
    # weight is strictly WORSE than doing nothing.
    #
    # This defaulted to index 3 for piper_arm's mechanically-locked joint 4,
    # and that dataset-specific default leaked into every LIBERO run. LIBERO's
    # dim 3 has std 0.0392 -- 62% of dim 4's and 50% of dim 5's, an ordinary
    # rotation axis -- so it was fed marginal noise for entire runs while the
    # log said only "Locking action dim 3".
    act_std = np.asarray(combined_stats["action"]["std"], dtype=float).reshape(-1)
    widest = float(act_std.max()) if act_std.size else 1.0
    # A dim varying <0.1% of the widest is a point mass in practice. piper_arm's
    # joint 4 is exactly 0; LIBERO's dim 3 sits 39x above this line.
    degenerate = [i for i in range(min(action_dim, act_std.size))
                  if act_std[i] <= 1e-3 * widest]

    if lock_joint_index is None:
        locked, why = degenerate, "the data (std is a point mass)"
    else:
        locked = [lock_joint_index] if 0 <= lock_joint_index < action_dim else []
        why = f"--lock_joint_index {lock_joint_index}"

    action_dim_weights = [1.0] * action_dim
    for i in locked:
        action_dim_weights[i] = 0.0

    # Always print the stds. The old message named the locked index and nothing
    # else, so there was no way to tell a mechanically-dead joint from a live
    # one being silently randomised.
    print("Action dim std: " + "  ".join(
        f"[{i}]{act_std[i]:.4f}{'*' if i in locked else ''}"
        for i in range(min(action_dim, act_std.size))))
    if locked:
        print(f"Locked action dims {locked} (weight=0) from {why}; "
              f"action_dim_weights={action_dim_weights}")
        for i in locked:
            if i not in degenerate:
                print(f"  [WARN] dim {i} std {act_std[i]:.4f} is "
                      f"{100 * act_std[i] / widest:.1f}% of the widest dim -- it is "
                      f"NOT constant. Weight 0 makes the model SAMPLE this dim "
                      f"from its marginal every step, which is worse than "
                      f"emitting its mean. Drop --lock_joint_index unless the "
                      f"joint is mechanically dead.")
    else:
        print(f"All {action_dim} action dims weighted equally; "
              f"action_dim_weights={action_dim_weights}")

    if paraphrase_augment:
        # Preflight, not a runtime warning. A sentence with no written variants
        # trains UNAUGMENTED while the rest vary, so the model keeps surface
        # form as a usable key for exactly those tasks -- and the run cannot
        # answer whether augmentation works. Too long a run to find that out
        # from the eval.
        from libero_paraphrase import coverage, instruction_strings, load_table
        # Union over every dataset: with several --dataset_id the task lists
        # differ, and an instruction that only appears in the second one still
        # needs variants.
        instructions, seen = [], set()
        for did in dataset_ids:
            raw = getattr(metas[did], "tasks", None)
            if raw is None:
                continue
            for ins in instruction_strings(raw):
                key = " ".join(str(ins).split())
                if key not in seen:
                    seen.add(key)
                    instructions.append(key)
        if not instructions:
            print("[paraphrase] dataset metadata exposes no task list; coverage "
                  "cannot be checked here. Run\n"
                  "  python -m libero_paraphrase --dataset_id <id> --min_variants N\n"
                  "before trusting this run.")
        else:
            table, under = coverage(
                instructions, paraphrase_limit, paraphrase_min_variants,
                load_table(paraphrase_file) if paraphrase_file else None)
            sizes = sorted(len(v) for v in table.values())
            print(f"[paraphrase] {len(table)} instructions, "
                  f"{len(table) - len(under)} at >= {paraphrase_min_variants} "
                  f"variants (min {sizes[0]}, median {sizes[len(sizes) // 2]}, "
                  f"max {sizes[-1]})")
            if under:
                shown = "\n".join(f"    {len(table[k]):>2}  {k}" for k in under[:12])
                raise SystemExit(
                    f"[paraphrase] {len(under)} instruction(s) below "
                    f"--paraphrase_min_variants {paraphrase_min_variants}:\n"
                    f"{shown}\n"
                    f"{'    ...' if len(under) > 12 else ''}\n"
                    f"  Write a table for these and pass --paraphrase_file:\n"
                    f"    python -m libero_paraphrase --dataset_id "
                    f"{dataset_ids[0]} --out para.json\n"
                    f"  then hand-edit the entries listed as UNDER. Lower "
                    f"--paraphrase_min_variants only if you accept that those "
                    f"tasks train unaugmented.")

    # LoRA sizing. alpha defaults to 2x rank because LoRALinear scales by
    # alpha/rank, and the shipped pair is 32/16 = 2.0 -- holding alpha fixed
    # while raising rank would quietly HALVE the adapter's effective strength,
    # moving two variables when only one was asked for.
    lora_kw: dict = {}
    if lora_rank is not None:
        lora_kw["lora_rank"] = int(lora_rank)
        lora_kw["lora_alpha"] = float(lora_alpha if lora_alpha is not None
                                      else 2.0 * int(lora_rank))
    elif lora_alpha is not None:
        lora_kw["lora_alpha"] = float(lora_alpha)
    if vision_lora_num_layers is not None:
        lora_kw["vision_lora_num_layers"] = int(vision_lora_num_layers)

    _awr_paths = ([awr_rewards] if isinstance(awr_rewards, str)
                  else list(awr_rewards or []))
    _awr_paths = [q for q in _awr_paths if q]
    _awr_idx = ([awr_dataset_index] if isinstance(awr_dataset_index, int)
                else list(awr_dataset_index or []))
    if len(_awr_paths) > 1 and len(_awr_idx) == 1 and _awr_idx == [0]:
        raise ValueError(
            f"{len(_awr_paths)} sidecars but --awr_dataset_index left at its "
            f"default: say which --dataset_id position each one describes, "
            f"e.g. --awr_dataset_index 0 1 2.")
    awr = (AWRWeightSet(_awr_paths, _awr_idx[:len(_awr_paths)],
                        awr_beta, awr_clip, awr_reward, group_by=awr_group_by,
                        drop_failures=awr_drop_failures,
                        min_weight=awr_min_weight)
           if _awr_paths else None)
    if awr is not None and len(dataset_ids) > 1:
        # Mixing the demo set in is the right defence against the corpus being
        # narrow -- AWR trains on the collected suite alone, so without demos
        # the model only ever sees that suite and the others drift. What used to
        # make it unsafe is that the join is on episode_index, which is
        # dataset-LOCAL: ConcatDataset does not renumber, so the corpus's weight
        # for episode 5 would also land on the demo set's episode 5, silently.
        #
        # _TagDataset now stamps dataset_index on every sample and lookup()
        # keys on the pair, so only the dataset the sidecar describes is
        # reweighted and everything else gets 1.0. All that is left to get
        # wrong is WHICH position the sidecar describes.
        bad = [i for i in awr.indices if not (0 <= i < len(dataset_ids))]
        if bad:
            raise ValueError(
                f"--awr_dataset_index {bad} out of range for "
                f"{len(dataset_ids)} datasets.")
        unweighted = [d for k, d in enumerate(dataset_ids) if k not in awr.indices]
        print(f"AWR mixing: weights apply to --dataset_id position(s) "
              f"{awr.indices} "
              f"({', '.join(dataset_ids[i] for i in awr.indices)}).")
        print(f"  weight 1.0 (unreweighted): "
              f"{', '.join(unweighted) if unweighted else '<none>'}")
        if not unweighted:
            print("  [WARN] every dataset is a collected corpus -- nothing here "
                  "holds the policy to the demonstrations, which is the whole "
                  "reason mixing exists.")
        print("  The mix ratio is set by their relative SIZES (the sampler is "
              "uniform over the concatenation), so read the frame counts below "
              "against how much anti-forgetting pressure you want.")

    # ---- state-noise augmentation: resolve the mode and SAY what it does ----
    # This has been invisible since it was written: one hardcoded 0.01 on a
    # state vector whose dims span 65x in natural scale. Print the realised
    # per-dim strength so it can never be invisible again.
    _ss = (combined_stats.get("observation.state") or {}).get("std")
    state_std_t = None if _ss is None else torch.as_tensor(
        np.asarray(_ss, dtype=np.float32).reshape(-1))
    if state_noise_frac > 0.0 and state_std_t is None:
        raise ValueError("--state_noise_frac needs observation.state stats to "
                         "scale by, and none were found in the dataset "
                         "metadata. Use --state_noise_abs instead.")
    if state_noise_prob > 0.0 and (state_noise_frac > 0.0 or state_noise_abs > 0.0):
        _mode = ("per-dim, frac %.4g of each dim's own std" % state_noise_frac
                 if state_noise_frac > 0.0
                 else "ABSOLUTE, sigma %.4g in RAW units" % state_noise_abs)
        print(f"State-noise augmentation: {_mode}, p={state_noise_prob:g} "
              f"(applied BEFORE normalisation)")
        if state_std_t is not None:
            eff = [(state_noise_frac if state_noise_frac > 0.0
                    else state_noise_abs / max(float(v), 1e-12))
                   for v in state_std_t.tolist()]
            print("  realised noise / that dim's own std: " + "  ".join(
                f"[{i}]{100 * e:.1f}%" for i, e in enumerate(eff)))
            worst = max(range(len(eff)), key=lambda i: eff[i])
            if eff[worst] > 0.25:
                print(f"  [WARN] dim {worst} receives {100 * eff[worst]:.0f}% of its "
                      f"own std. On LIBERO dims 6/7 are the gripper finger joints "
                      f"(std ~0.014 m), i.e. the 'am I holding it' channel. "
                      f"--state_noise_frac scales per dim instead of flattening "
                      f"a 65x range onto one number.")
    else:
        print("State-noise augmentation: OFF")

    # Preflight. Each of these has a silent-wrong-run failure mode: the flag is
    # accepted, training completes, and the result answers a different question
    # than the one asked.
    if vision_token_source == "resnet":
        side = int(resnet_tokens ** 0.5)
        px = resnet_input_size / max(side, 1)
        print(f"Vision tokens: ResNet-18 truncated at layer3 (trainable, 3.0M) — "
              f"{resnet_tokens} tok @ {resnet_input_size}px "
              f"= {px:.1f} px/token")
        if px > 32.0:
            print(f"  [WARN] {px:.1f} px/token is COARSER than the frozen VLM's 32 "
                  f"px merged patches. The CNN exists for the precision the ViT "
                  f"cannot reach; at this grid it is running below the backbone it "
                  f"is meant to sharpen. Raise --resnet_tokens.")
        # Per-camera fine grid. The native map is input_size/16 (ResNet-18 cut at
        # layer3), so out_tokens above (input/16)^2 upsamples an already-pooled
        # map and buys nothing; at the native value the read is 1:1.
        native = (resnet_input_size // 16) ** 2
        print(f"  native feature map = {resnet_input_size}//16 squared = {native} "
              f"tok; resnet_tokens={resnet_tokens} "
              f"({'1:1' if resnet_tokens == native else 'pooled from ' + str(native)})")
        fine_cams = list(resnet_fine_cameras or [])
        if resnet_fine_tokens > 0 or fine_cams:
            if resnet_fine_tokens <= 0 or not fine_cams:
                raise ValueError(
                    "--resnet_fine_cameras and --resnet_fine_tokens must be given "
                    "together; one without the other is a no-op that completes a "
                    f"full run (got cameras={fine_cams}, tokens={resnet_fine_tokens}).")
            _fs = int(resnet_fine_tokens ** 0.5)
            if _fs * _fs != resnet_fine_tokens:
                raise ValueError(f"--resnet_fine_tokens must be a perfect square "
                                 f"for avg pooling, got {resnet_fine_tokens}")
            # A name that is not a real camera key falls through to resnet_tokens
            # inside _resnet_tokens, silently. That costs a whole run.
            unknown = [c for c in fine_cams if c not in camera_keys]
            if unknown:
                raise ValueError(
                    f"--resnet_fine_cameras {unknown} are not cameras in this "
                    f"dataset. Available: {camera_keys}. Left unchecked these "
                    f"fall back to --resnet_tokens with no warning, and the fine "
                    f"grid never runs.")
            if resnet_fine_tokens > native:
                print(f"  [WARN] --resnet_fine_tokens {resnet_fine_tokens} exceeds "
                      f"the native {native}; this upsamples a pooled map and adds "
                      f"sequence length for no new information.")
            n_fine, n_coarse = len(fine_cams), len(camera_keys) - len(fine_cams)
            total = n_fine * resnet_fine_tokens + n_coarse * resnet_tokens
            print(f"  fine grid: {fine_cams} -> {resnet_fine_tokens} tok "
                  f"({resnet_input_size / max(int(resnet_fine_tokens ** 0.5), 1):.1f} "
                  f"px/token); others -> {resnet_tokens} tok")
            # Count the STATE tokens rather than assuming one. --n_obs_steps
            # alone is inert (wilro_moe slices state_tok[:, -1:] unless
            # use_state_history), and nothing else in the startup log says
            # whether the temporal pathway is on -- so this line was the only
            # place it could have shown, and it hardcoded 1.
            n_state = obs if use_state_history else 1
            print(f"  vision tokens in the DiT sequence: {total} "
                  f"(was {len(camera_keys) * resnet_tokens} at a uniform grid); "
                  f"state tokens: {n_state}"
                  + (f" (n_obs_steps={obs}, use_state_history ON -- the model "
                     f"can see velocity)" if use_state_history else
                     (f" (n_obs_steps={obs} FETCHED but sliced to the last "
                      f"frame: --use_state_history is OFF, so the extra frames "
                      f"are encoded and discarded)" if obs > 1 else ""))
                  + f"\n  -> DiT sequence length {1 + n_state + total + 64}")

    # Build wilro config
    # SmolVLM2-500M's tower is pretrained at 512 with patch 16 and pixel-shuffle
    # 4, so tokens per camera = (v/16/4)^2: 36 at the shipped 384, 64 at native.
    # 384 is BELOW native, not a safe default.
    if vision_input_size % 64:
        raise SystemExit(
            f"--vision_input_size {vision_input_size} must be divisible by 64 "
            f"(patch 16 x pixel-shuffle 4). {vision_input_size // 16} patches "
            f"per side is not divisible by the shuffle factor of 4.")
    _vtok = (vision_input_size // 64) ** 2
    print(f"[wilro_moe] VLM tower at {vision_input_size}px -> "
          f"{(vision_input_size // 16)}^2 patches -> {_vtok} tokens/camera"
          + ("  (SmolVLM2-500M's NATIVE resolution)" if vision_input_size == 512
             else f"  (native is 512 = 64 tokens; this is "
                  f"{'below' if vision_input_size < 512 else 'above'} it)"))

    cfg = WilroMoEConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=obs,
        horizon=horizon,
        n_action_steps=n_action_steps,
        state_dim=state_dim,
        action_dim=action_dim,
        num_vlm_layers=16,  # DiT depth = number of VLM KV pairs consumed
        kv_capture_strategy=kv_capture_strategy,
        kv_capture_layers=kv_capture_layers or [],
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
        action_dim_weights=action_dim_weights,
        # n_action_steps == horizon → no exponential decay needed.
        pos_decay_lambda=0.0,
        contrastive_loss_weight=contrastive_loss_weight,
        contrastive_margin=contrastive_margin,
        contrastive_hard_negatives=contrastive_hard_negatives,
        noise_temporal_correlation=noise_temporal_correlation,
        gripper_phase_weight=gripper_phase_weight,
        gripper_transition_window=int(gripper_transition_window),
        gripper_transition_thresh=float(gripper_transition_thresh),
        gripper_action_index=action_dim - 1,  # LIBERO OSC: gripper is the last dim
        time_sampling=time_sampling,
        time_lognormal_mean=time_lognormal_mean,
        time_lognormal_std=time_lognormal_std,
        **({} if lr is None else {"optimizer_lr": float(lr)}),
        **({} if warmup_steps is None else {"scheduler_warmup_steps": int(warmup_steps)}),
        **lora_kw,
        paraphrase_augment=paraphrase_augment,
        paraphrase_limit=paraphrase_limit,
        paraphrase_file=paraphrase_file,
        paraphrase_min_variants=paraphrase_min_variants,
        vision_token_source=vision_token_source,
        resnet_tokens=resnet_tokens,
        resnet_fine_cameras=list(resnet_fine_cameras or []),
        resnet_fine_tokens=int(resnet_fine_tokens),
        resnet_input_size=resnet_input_size,
        vision_input_size=vision_input_size,
        resnet_pool=resnet_pool,
        use_state_history=use_state_history,
        resnet_motion_tokens=resnet_motion_tokens,
        resnet_motion_stride=resnet_motion_stride,
        num_experts=num_experts,
        expert_num_layers=expert_num_layers,
        dit_hidden_size=dit_hidden_size,
        vlm_capture_layers=[int(t) for t in vlm_capture_layers.split(",") if t.strip()],
        resnet_expert_adapter_dim=resnet_expert_adapter_dim,
        router_temperature=router_temperature,
        router_top_k=router_top_k,
        router_balance_weight=router_balance_weight,
    )

    # Model + checkpoint loading
    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        policy = WilroMoEPolicy(cfg)

        ckpt_path = Path(resume_from_checkpoint)
        if ckpt_path.exists():
            local_ckpt_path = ckpt_path
            print(f"Using local checkpoint: {local_ckpt_path}")
        else:
            print(f"Local path not found, downloading from HuggingFace Hub: {resume_from_checkpoint}")
            local_ckpt_path = Path(huggingface_hub.snapshot_download(resume_from_checkpoint))

        model_file = local_ckpt_path / "model.safetensors"
        if not model_file.exists():
            candidates = list(local_ckpt_path.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No .safetensors file found in {local_ckpt_path}")
            model_file = candidates[0]

        step, epoch = 0, 0
        saved_cfg_json = {}
        for config_name in ("config.json", "pretrained_config.json"):
            config_file = local_ckpt_path / config_name
            if config_file.exists():
                with open(config_file) as f:
                    saved_cfg_json = json.load(f)
                step = saved_cfg_json.get("training_step", 0)
                epoch = saved_cfg_json.get("training_epoch", 0)
                saved_total = saved_cfg_json.get("training_steps_total", 0)
                # An explicit --training_steps wins over the checkpoint's. The
                # opposite order is how --lr was silently thrown away on every
                # resume in this repo (4caed2d); the schedule is worth even
                # more than the peak LR, since it decides whether the run
                # anneals at all.
                if steps_cli is not None and saved_total > 0 and saved_total != steps_cli:
                    print(f"--training_steps {steps_cli} OVERRIDES the "
                          f"checkpoint's {saved_total}; the cosine schedule is "
                          f"rebuilt over {steps_cli} and the LR at step {step} "
                          f"will differ from the original run's.")
                elif saved_total > 0:
                    training_steps = saved_total
                print(f"Read config from {config_file.name}: step={step}, epoch={epoch}, training_steps_total={training_steps}")
                # A changed VLM resolution LOADS fine -- proj is per-token and the
                # experts cross-attend over a variable-length KV, so no shape
                # mismatches. What changes is the statistics the cross-attention
                # was fitted to, and re-adapting costs LR that a late cosine does
                # not have.
                _sv = saved_cfg_json.get("vision_input_size")
                if _sv is not None and int(_sv) != int(vision_input_size):
                    print(f"\n  !! RESUMING WITH A CHANGED VLM RESOLUTION: "
                          f"{_sv} -> {vision_input_size}  "
                          f"({(int(_sv)//64)**2} -> {(vision_input_size//64)**2} "
                          f"tokens/camera)\n"
                          f"     Nothing mismatches in shape, so this will NOT "
                          f"error -- it will quietly run the experts against "
                          f"cross-attention statistics they were not trained on.\n"
                          f"     Read the 'Scheduler fast-forwarded' LR below "
                          f"before trusting it: with no LR budget left the run "
                          f"cannot re-adapt and will finish degraded.\n",
                          flush=True)
                # Warn only on an ACTUAL geometry change. Passing --lora_rank 64
                # to continue a run that already trained at 64 is the normal way
                # to resume, and a warning there says the adapters are being
                # discarded when they load fine -- which is worth aborting over
                # if believed.
                changed = {k: (saved_cfg_json.get(k), v) for k, v in lora_kw.items()
                           if k in saved_cfg_json and saved_cfg_json[k] != v}
                if changed:
                    detail = ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in changed.items())
                    print(f"\n*** LoRA geometry CHANGED ({detail}). The "
                          f"checkpoint's adapters have different shapes, so they "
                          f"will be SKIPPED and the vision adapter restarts at "
                          f"zero -- the frozen base SigLIP, i.e. every bit of "
                          f"visual adaptation this checkpoint learned is "
                          f"discarded. Check 'Skipped N checkpoint keys' below. "
                          f"***\n")
                elif lora_kw:
                    print(f"LoRA geometry matches the checkpoint "
                          f"({', '.join(f'{k}={v}' for k, v in lora_kw.items())}); "
                          f"adapters will load.")
                break
        if step == 0 and local_ckpt_path.name.startswith("checkpoint-"):
            step = int(local_ckpt_path.name.split("-")[1])
        if start_step_override >= 0:
            # Loading weights from a run on a DIFFERENT dataset is not a
            # resume, and inheriting its step counter silently does three
            # things nobody asked for: the cosine is fast-forwarded, so the
            # new run starts mid-decay with no warmup; the step budget is
            # short by however far the old run got; and the progress bar
            # reports a number that belongs to another dataset.
            print(f"--start_step_override {start_step_override}: taking the "
                  f"WEIGHTS from step {step} but restarting the counter, so "
                  f"the schedule is rebuilt from scratch (warmup included) "
                  f"over {training_steps} steps. Epoch resets too "
                  f"(checkpoint said {epoch}) -- it counts passes over THIS "
                  f"dataset, and carrying the previous run's total over makes "
                  f"the saved training_epoch unreadable: a 7-epoch run on a "
                  f"new set was saved as 21 and read back as heavy "
                  f"overtraining that never happened.")
            step = int(start_step_override)
            epoch = 0
        print(f"Resuming from step {step}, epoch {epoch}")

        print(f"Loading weights from: {model_file}")
        # CPU, not device. Loading the file straight onto the GPU makes the
        # whole checkpoint resident ALONGSIDE the model that is about to receive
        # it -- roughly a second copy of every parameter, for the duration of
        # the copy. load_state_dict moves each tensor to its parameter's device
        # anyway, so staging on the host costs nothing and is why resuming needs
        # no more memory than starting fresh.
        ckpt_state = load_safetensors(model_file, device="cpu")

        policy.train()
        policy.to(device)
        cur_state = policy.state_dict()
        filtered = {k: v for k, v in ckpt_state.items() if k in cur_state and cur_state[k].shape == v.shape}

        skipped_ckpt = [k for k in ckpt_state if k not in filtered]
        missing_from_ckpt = [k for k in cur_state if k not in ckpt_state]
        if skipped_ckpt:
            print(f"Skipped {len(skipped_ckpt)} checkpoint keys (shape mismatch / removed): {skipped_ckpt[:10]}")
        if missing_from_ckpt:
            print(f"Missing {len(missing_from_ckpt)} keys not in checkpoint (will use init values): {missing_from_ckpt[:10]}")
        policy.load_state_dict(filtered, strict=False)
        n_loaded, n_file = len(filtered), len(ckpt_state)
        del ckpt_state, filtered, cur_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Loaded {n_loaded}/{len(policy.state_dict())} model keys from checkpoint ({n_file} keys in file)")

        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            dataset_stats=combined_stats,
        )

        # The cosine scheduler's base LR must be the PEAK (pre-decay) value: the
        # decay is reconstructed purely by fast-forwarding scheduler.step() `step`
        # times below. The checkpoint's saved "optimizer_lr" is the ALREADY-DECAYED
        # lr (overwritten at save time), so using it as the base double-applies the
        # decay → peak·cos(step)². Use the config peak (cfg.optimizer_lr — not in
        # the WilroMoEConfig kwargs, so it's the default peak) so fast-forwarding
        # rebuilds the correct peak·cos(step). Matches train_community.py.
        base_lr = cfg.optimizer_lr
        resume_warmup = saved_cfg_json.get("scheduler_warmup_steps", cfg.scheduler_warmup_steps)
        print(f"Scheduler base (peak) LR: {base_lr:.2e}  (decay rebuilt by "
              f"fast-forwarding to step {step})")

        trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=base_lr, weight_decay=cfg.optimizer_weight_decay)
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer_state_path = local_ckpt_path / "optimizer_state.pth"
        if optimizer_state_path.exists():
            try:
                # map_location="cpu": Optimizer.load_state_dict casts each state
                # tensor to its own parameter's device, so the GPU never holds
                # the loaded dict AND the optimizer's copy at once. Adam state is
                # 2 x trainable x 4 bytes -- 5.2 GB on wilro_moe -- so the
                # transient double is what pushes a resume into OOM at a batch
                # size that trains fine from scratch.
                optimizer.load_state_dict(
                    torch.load(optimizer_state_path, map_location="cpu"))
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr
                    param_group['initial_lr'] = base_lr
                print(f"Optimizer state loaded. Scheduler base LR set to peak {base_lr:.2e}")
            except ValueError as e:
                print(f"Skipping optimizer state — architecture mismatch ({e})")

        warmup_steps = resume_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )
        for _ in range(step):
            scheduler.step()
        print(f"Scheduler fast-forwarded to step {step}, LR = {optimizer.param_groups[0]['lr']:.2e}")
    else:
        policy = WilroMoEPolicy(cfg)
        policy.train()
        policy.to(device)

        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            dataset_stats=combined_stats,
        )
        step = 0
        epoch = 0

        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        n_frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}  "
              f"(frozen: {n_frozen:,})")

        fresh_lr = cfg.optimizer_lr
        fresh_warmup = cfg.scheduler_warmup_steps
        optimizer = torch.optim.Adam(trainable_params, lr=fresh_lr, weight_decay=cfg.optimizer_weight_decay)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=fresh_warmup,
            num_training_steps=training_steps,
        )

    # Optional DiT gradient checkpointing (frozen VLM is unaffected — it runs in no_grad).
    if gradient_checkpointing and hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable()

    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # Dataset setup — read fps from metadata instead of hardcoding. piper_arm
    # is 30 fps but libero / community datasets are commonly 10 fps; using a
    # mismatched frame_time makes every requested delta_timestamp fall outside
    # tolerance_s and the constructor raises. All datasets must share one fps so
    # the action horizon means the same real time everywhere.
    fps = int(getattr(ref_meta, "fps", 30) or 30)
    for did in dataset_ids[1:]:
        f2 = int(getattr(metas[did], "fps", fps) or fps)
        if f2 != fps:
            raise ValueError(
                f"Dataset '{did}' fps={f2} differs from '{dataset_ids[0]}' fps={fps}. "
                f"Resample to a common fps before mixing — the chunk horizon must "
                f"cover the same real time across datasets."
            )
    frame_time = 1 / fps
    print(f"Dataset fps: {fps} (frame_time={frame_time:.4f}s)")

    # Observation window: last `obs` frames ending at t=0
    obs_temporal_window = [-i * frame_time for i in range(obs)][::-1]
    # Action window: `horizon` steps starting at t=0
    action_temporal_window = [i * frame_time for i in range(horizon)]

    # Cameras get ONE frame unless the ResNet motion path is on. The VLM reads
    # imgs[:, -1] either way and is 40.8% of step time, so a second frame is
    # deliberately NOT bought for it: over 100ms the semantics do not change,
    # only the motion, and motion is what the ResNet feature-map diff extracts.
    # The second frame also doubles per-camera video decode, and these workers
    # have already been SIGKILLed at 5.3 GB of decoded frames in flight -- drop
    # --num_workers if this run starts dying around step 200.
    if resnet_motion_tokens > 0:
        cam_window = [-resnet_motion_stride * frame_time, 0.0]
        print(f"Camera window: {cam_window} ({resnet_motion_stride} frame(s) back "
              f"= {resnet_motion_stride * frame_time * 1000:.0f}ms) — ResNet motion path")
    else:
        cam_window = [0.0]

    delta_timestamps = {
        "observation.state": obs_temporal_window,
        "action": action_temporal_window,
        **{key: cam_window for key in camera_keys},
    }

    # `tolerance_s` must accommodate the dataset's frame interval — too tight
    # and every delta lookup raises. Half a frame is a safe upper bound.
    tolerance_s = max(0.005, frame_time / 2)

    # Build each dataset, concatenate, and accumulate episode boundaries in the
    # concatenated index space (optionally filtered per-dataset by --max_episode_index).
    # One resize for every dataset, or none at all. Applying it to only the
    # odd one out would leave two different preprocessing paths feeding one
    # model, and the difference would be invisible in the loss.
    all_shapes = {sh for d in vis_shapes.values() for sh in d.values()}
    resize_to = None
    if len(all_shapes) > 1:
        non_square = [sh for sh in all_shapes if sh[-1] != sh[-2]]
        if non_square:
            raise ValueError(
                f"cameras differ in resolution AND some are not square "
                f"({sorted(all_shapes)}). The model pads non-square frames to "
                f"square before resizing, so resizing them here would change "
                f"the aspect handling. Convert them to a common size first.")
        resize_to = int(cfg.vision_input_size)
        print(f"\nCamera resolutions differ across datasets: "
              f"{ {d: sorted(set(v.values())) for d, v in vis_shapes.items()} }\n"
              f"  -> resizing every frame to {resize_to}x{resize_to} at LOAD "
              f"time. ConcatDataset stacks raw tensors, so mixed resolutions "
              f"would fail in collate; the model resizes to vision_input_size "
              f"anyway, so doing it here changes nothing but the timing.")
    if load_image_size:
        # The model pads to square and interpolates every frame to
        # vision_input_size anyway, so doing it in the workers costs nothing at
        # the model and cuts what the loader must hold, pin and copy by
        # (size/native)^2. A 480 source at 384 is 0.64x. This is the difference
        # between a batch of 64 costing 354 MB and 226 MB, per in-flight batch,
        # per worker.
        #
        # Opt-in rather than automatic: v2.Resize antialiases and the model's
        # F.interpolate does not, so enabling it changes the pixels slightly
        # and a run started with it is not bit-comparable to one without.
        if resize_to and resize_to != int(load_image_size):
            print(f"  --load_image_size {load_image_size} overrides the "
                  f"{resize_to} chosen to reconcile mixed resolutions.")
        resize_to = int(load_image_size)
    img_tf = v2.Resize((resize_to, resize_to), antialias=True) if resize_to else None

    sub_datasets = []
    ep_from: list[int] = []
    ep_to: list[int] = []
    ep_ds: list[str] = []          # which dataset each episode came from
    offset = 0
    first_root = None
    for did in dataset_ids:
        _t = time.time()
        print(f"[data] opening {did}"
              + ("  (syncing cache; first pull can take a long time and is "
                 "silent unless --download_progress)" if cache_sync else ""),
              flush=True)
        ds = LeRobotDataset(
            did, delta_timestamps=delta_timestamps,
            force_cache_sync=cache_sync, revision="main", tolerance_s=tolerance_s,
            image_transforms=img_tf,
            **({} if video_backend is None else {"video_backend": video_backend}),
        )
        print(f"[data]   ...{time.time() - _t:.0f}s", flush=True)
        if first_root is None:
            first_root = ds.root
        # Episode spans must tile the table. delta_timestamps make
        # _get_query_indices clamp every lookup to [dataset_from_index,
        # dataset_to_index - 1], and it is the ONLY consumer of those columns:
        # a set whose offsets are wrong loads, indexes and prints correctly,
        # then raises from inside a DataLoader worker on the first batch --
        #   IndexError: Invalid key: 863104 is out of bounds for size 575101
        # Published VLABench ships dataset_from_index = length * episode_index
        # instead of a running sum, so this is not hypothetical.
        E = ds.meta.episodes
        fr = np.asarray(E["dataset_from_index"], dtype=np.int64)
        to = np.asarray(E["dataset_to_index"], dtype=np.int64)
        o = np.argsort(np.asarray(E["episode_index"], dtype=np.int64))
        fr, to = fr[o], to[o]
        n_rows_ds = len(ds.hf_dataset)
        if fr[0] != 0 or to[-1] != n_rows_ds or not (to[:-1] == fr[1:]).all():
            raise ValueError(
                f"Dataset '{did}': meta/episodes row offsets do not tile the "
                f"table.\n"
                f"  spans cover [{fr[0]}, {to[-1]}), table has {n_rows_ds} rows; "
                f"{int((to[:-1] != fr[1:]).sum())} of {len(fr) - 1} boundaries "
                f"gap or overlap.\n"
                f"  Training would fail on its first batch inside a DataLoader "
                f"worker with an out-of-bounds IndexError, which does not name "
                f"this as the cause.\n"
                f"  For VLABench, src/convert_vlabench_to_libero.py rebuilds "
                f"these from the data.")
        ep_ids = np.array(ds.hf_dataset["episode_index"])
        changes = np.where(np.diff(ep_ids) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(ep_ids)]])
        kept = 0
        for s, e in zip(starts, ends):
            if max_episode_index is not None and int(ep_ids[s]) > max_episode_index:
                continue
            ep_from.append(offset + int(s))
            ep_to.append(offset + int(e))
            ep_ds.append(did)
            kept += 1
        suffix = f" (<= ep {max_episode_index})" if max_episode_index is not None else ""
        print(f"  {did}: {len(ds)} frames, {kept} episodes{suffix}")
        sub_datasets.append(_TagDataset(ds, len(sub_datasets)))
        offset += len(ds)

    dataset = ConcatDataset(sub_datasets)
    print(fingerprint_line(), flush=True)
    print(f"Combined dataset: {len(dataset)} frames, {len(ep_from)} episodes "
          f"across {len(sub_datasets)} dataset(s)")

    # Gripper-transition preflight. The mask is built from |delta gripper| over
    # the action chunk, so a dataset whose gripper RAMPS instead of flipping
    # makes the threshold catch only the steepest frame of the ramp and the
    # window then dilates around several wrong centres -- silently, since the
    # loss still runs. Sample a few hundred chunks and say which case this is.
    if gripper_phase_weight != 1.0:
        import random as _rnd
        _gi = action_dim - 1
        _r = _rnd.Random(0)
        _dg = []
        _runs, _with = [], 0
        # dataset[i] decodes this sample's video frames too, so keep the count
        # modest -- this is a preflight, not a statistic.
        _n = min(150, len(dataset))
        print(f"  gripper phase: sampling {_n} chunks to check transition "
              f"detection...", end="", flush=True)
        for _i in _r.sample(range(len(dataset)), _n):
            _a = dataset[_i].get("action")
            if _a is None or _a.ndim != 2:
                continue
            _g = _a[:, _gi].float()
            _d = (_g[1:] - _g[:-1]).abs()
            _dg.append(_d)
            _t = (_d > gripper_transition_thresh)
            _with += int(bool(_t.any()))
            _run = 0
            for _v in _t.tolist():
                if _v:
                    _run += 1
                elif _run:
                    _runs.append(_run); _run = 0
            if _run:
                _runs.append(_run)
        if _dg:
            _all = torch.cat(_dg)
            _q = [float(_all.quantile(q)) for q in (0.5, 0.95, 0.99)]
            _flag = float((_all > gripper_transition_thresh).float().mean())
            _w = int(gripper_transition_window)
            print(f"\r  gripper phase: weight {gripper_phase_weight}, window +/-{_w} "
                  f"= {2 * _w + 1} positions ({(2 * _w + 1) / fps:.1f}s), "
                  f"thresh {gripper_transition_thresh}")
            print(f"    |d gripper| over {_n} chunks: p50={_q[0]:.3f} "
                  f"p95={_q[1]:.3f} p99={_q[2]:.3f}   flagged {_flag * 100:.2f}% "
                  f"of positions, {_with / max(_n, 1) * 100:.0f}% of chunks have one")
            if _runs:
                _one = sum(1 for r in _runs if r == 1) / len(_runs)
                if _one > 0.9:
                    print(f"    run length 1 in {_one * 100:.0f}% of transitions -- "
                          f"the gripper FLIPS in a single step and the threshold "
                          f"sees it cleanly.")
                else:
                    print(f"    [WARN] only {_one * 100:.0f}% of transitions are a "
                          f"single step: this gripper RAMPS. The threshold is "
                          f"catching part of the ramp and the window is dilating "
                          f"around several centres -- retune "
                          f"--gripper_transition_thresh before trusting the "
                          f"window.")
            if _flag == 0.0:
                print(f"    [WARN] the threshold flags NOTHING. "
                      f"--gripper_phase_weight {gripper_phase_weight} is inert.")

    # Build task_index → description mapping from the first dataset's tasks.parquet.
    # Batches carry the per-frame "task" string directly (preferred by the loop);
    # for multi-dataset, task_index is dataset-local so we rely on batch["task"].
    task_idx_to_description: dict[int, str] = {}
    try:
        tasks_parquet_path = first_root / "meta" / "tasks.parquet"
        if tasks_parquet_path.exists():
            tasks_df = pd.read_parquet(tasks_parquet_path)
            if "task_index" in tasks_df.columns:
                task_idx_to_description = {
                    int(row["task_index"]): str(idx)
                    for idx, row in tasks_df.iterrows()
                }
            print(f"Loaded {len(task_idx_to_description)} task descriptions from tasks.parquet:")
            for idx, desc in task_idx_to_description.items():
                print(f"  [{idx}] {desc}")
        else:
            print("tasks.parquet not found; task_description will not be added to batches.")
    except Exception as e:
        print(f"Warning: could not load tasks.parquet: {e}")

    if combined_stats and "observation.state" in combined_stats:
        s = combined_stats["observation.state"]
        print(f"\nNorm stats observation.state:")
        print(f"  mean={s.get('mean', 'N/A')}")
        print(f"  std ={s.get('std',  'N/A')}")
    else:
        print("WARNING: observation.state not found in combined_stats — will not be normalized!")
    if combined_stats and "action" in combined_stats:
        s = combined_stats["action"]
        print(f"Norm stats action:")
        print(f"  mean={s.get('mean', 'N/A')}")
        print(f"  std ={s.get('std',  'N/A')}")
    # ---- held-out split, by EPISODE ----
    # Splitting by frame would put frames from one episode on both sides: the
    # neighbouring frame is nearly the same image with nearly the same action,
    # so a held-out loss built that way measures interpolation and reports a
    # gap of roughly zero no matter how badly the model has memorised.
    val_ep_idx: list = []
    if val_episodes > 0:
        rng = np.random.default_rng(seed=42)
        by_ds: dict = {}
        for i, d in enumerate(ep_ds):
            by_ds.setdefault(d, []).append(i)
        # Proportional per dataset, so a mixed run does not hold out only the
        # big one and then report a number about the wrong domain.
        total = len(ep_ds)
        for d, idxs in by_ds.items():
            k = max(1, round(val_episodes * len(idxs) / total))
            k = min(k, max(0, len(idxs) - 1))
            if k:
                val_ep_idx += list(rng.choice(idxs, size=k, replace=False))
    val_set = set(val_ep_idx)
    tr_idx = [i for i in range(len(ep_ds)) if i not in val_set]

    # The `fit` slice must be drawn the SAME way the held-out set was, from the
    # same rng, or the two columns are not measuring the same thing. Taking
    # tr_idx[:n] instead looks harmless and is not: lerobot/libero is ordered by
    # SUITE, so the first n training episodes are all one suite while the
    # held-out set spans all forty tasks. The difference then reports
    # "suite A vs everything" on top of "trained vs held out", and on a real run
    # that produced a gap of +400% while held-out itself was flat.
    fit_ep_idx: list = []
    if val_ep_idx:
        by_ds_tr: dict = {}
        for i in tr_idx:
            by_ds_tr.setdefault(ep_ds[i], []).append(i)
        want: dict = {}
        for i in val_ep_idx:
            want[ep_ds[i]] = want.get(ep_ds[i], 0) + 1
        for d, k in want.items():
            pool = by_ds_tr.get(d, [])
            if pool:
                fit_ep_idx += list(rng.choice(pool, size=min(k, len(pool)),
                                              replace=False))

    def mk_sampler(idxs, shuffle):
        return EpisodeAwareSampler(
            dataset_from_indices=[ep_from[i] for i in idxs],
            dataset_to_indices=[ep_to[i] for i in idxs],
            drop_n_first_frames=0, drop_n_last_frames=0, shuffle=shuffle)

    def mk_loader(idxs, shuffle, workers):
        kw = {} if workers == 0 else {"prefetch_factor": max(1, int(prefetch_factor))}
        return torch.utils.data.DataLoader(
            dataset, num_workers=workers, batch_size=batch_size,
            sampler=mk_sampler(idxs, shuffle),
            pin_memory=device.type != "cpu", drop_last=True, **kw)

    sampler = mk_sampler(tr_idx, True)
    print(f"EpisodeAwareSampler: {len(sampler)} frames "
          f"over {len(tr_idx)} episodes")
    dataloader = mk_loader(tr_idx, True, num_workers)
    # Each worker forks a copy of the dataset, and Python refcounting
    # gradually un-shares the copy-on-write pages holding the Arrow table --
    # so host RAM climbs for hundreds of steps and then the runtime SIGINTs
    # the process. It reads as "^C" with no traceback, which looks nothing
    # like a memory error. _rss_gb below is there to make it visible.
    # Predict the in-flight cost instead of discovering it as a SIGKILL a few
    # hundred steps in. A killed worker surfaces as
    #   RuntimeError: DataLoader worker (pid ...) is killed by signal: Killed
    # raised from wherever the main process happened to be -- the traceback
    # points at the model, never at the loader.
    px = resize_to if resize_to else max(
        (sh[-1] for d in vis_shapes.values() for sh in d.values()), default=0)
    per_sample_mb = len(camera_keys) * 3 * px * px * 4 / 1048576.0
    inflight = per_sample_mb * batch_size * max(1, num_workers) * \
        max(1, int(prefetch_factor))
    print(f"DataLoader: {num_workers} worker(s), prefetch "
          f"{prefetch_factor}, batch {batch_size}, "
          f"{len(camera_keys)} cam @ {px or '?'}px\n"
          f"  ~{per_sample_mb:.1f} MB/sample -> ~{inflight / 1024:.1f} GB of "
          f"decoded frames in flight, plus an equal pinned copy"
          + ("   <-- large; lower --batch_size / --num_workers / "
             "--prefetch_factor, or pass --load_image_size 384"
             if inflight / 1024 > 3 else ""))

    val_loader = fit_loader = None
    if val_ep_idx:
        # shuffle=True on BOTH. val_max_batches x batch_size is far short of the
        # held-out set -- 20 x 60 = 1200 frames against ~6500 -- so an unshuffled
        # sampler scored the same arbitrary first ~7 episodes every pass and
        # called it the held-out loss. Whether those seven happened to be easy or
        # hard then set the absolute level for the whole run, which is how three
        # runs of the same model family came back with held-out at 0.25, 1.02 and
        # 1.30 while all three scored ~68% on the same held-out init states.
        # Shuffling spreads the same budget over all 40 episodes; run_eval_loss
        # pins the RNG, so successive passes still draw the SAME spread.
        val_loader = mk_loader(val_ep_idx, True, min(2, num_workers))
        # A same-sized slice of TRAINING episodes, scored the SAME way (eval
        # mode, no augmentation). Without it the only comparison available is
        # held-out against the running train loss, and those two differ by
        # dropout, image/state augmentation, paraphrase sampling AND the
        # contrastive term -- which is train-only here -- so their difference
        # is not a generalisation gap. fit vs held-out is.
        fit_loader = mk_loader(fit_ep_idx, True, min(2, num_workers))
        n_val_frames = sum(ep_to[i] - ep_from[i] for i in val_ep_idx)
        per_ds = {}
        for i in val_ep_idx:
            per_ds[ep_ds[i]] = per_ds.get(ep_ds[i], 0) + 1
        print(f"Validation: {len(val_ep_idx)} episodes held out "
              f"({n_val_frames} frames, {100 * n_val_frames / max(1, sum(ep_to[i] - ep_from[i] for i in range(len(ep_ds)))):.1f}%), "
              f"every {val_every} steps, <= {val_max_batches} batches\n"
              f"  per dataset: {per_ds}\n"
              f"  each pass scores {val_max_batches * batch_size} frames of "
              f"{n_val_frames} ({100 * val_max_batches * batch_size / max(1, n_val_frames):.0f}%), "
              f"sampled across all {len(val_ep_idx)} episodes"
              + ("  -- raise --val_max_batches for a steadier number"
                 if val_max_batches * batch_size < 0.5 * n_val_frames else ""))
    else:
        print("Validation: DISABLED (--val_episodes 0). Training loss alone "
              "cannot tell fitting from memorising.")

    @torch.no_grad()
    def run_eval_loss(loader):
        """Mean loss in EVAL mode with no augmentation.

        policy.eval() is what turns paraphrase sampling off (the model gates it
        on self.training), and it is also what drops the contrastive term, so
        both columns below are the same quantity measured on different
        episodes."""
        was_training = policy.training
        policy.eval()
        # Same t and same noise on every pass, and on BOTH loaders.
        # compute_loss draws a fresh flow timestep per sample and a fresh
        # source noise; without pinning them, two consecutive validations
        # differ by the draw as much as by the model. Measured on a real run:
        # the fit/held-out gap swung 8.3 -> 7.5 -> 22.2 -> 12.9 -> 21.8 -> 12.9
        # across adjacent passes while held-out itself moved by under 0.01.
        # Pinning also means `fit` and `held-out` are scored at the SAME
        # timesteps, so their difference is about the episodes and nothing else.
        cpu_state = torch.get_rng_state()
        cuda_state = (torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() else None)
        torch.manual_seed(20260829)
        tot, n = 0.0, 0
        # Magnitude probe. The flow loss says how far the predicted velocity
        # field is from the demo actions; it does not say in WHICH direction,
        # so a policy that has drifted toward larger, faster actions and one
        # that has simply got worse produce the same rising number. Sampling a
        # chunk on the first batch and comparing |a| against the target's
        # separates them: drift shows as a ratio moving away from 1.0 while the
        # loss rises, degradation shows as the loss rising with the ratio flat.
        amp_pred = amp_tgt = 0.0
        for i, b in enumerate(loader):
            if i >= val_max_batches:
                break
            for k in b:
                if isinstance(b[k], torch.Tensor):
                    b[k] = b[k].to(device, non_blocking=True)
            if "task" in b and isinstance(b["task"], (list, tuple)):
                b["task_description"] = b["task"]
            b = preprocessor(b)
            with (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                  if device.type == "cuda"
                  else torch.autocast(device_type="cpu", enabled=False)):
                loss, _ = policy.forward(b)
            tot += float(loss.detach()); n += 1
            if i == 0 and hasattr(policy, "predict_action_chunk"):
                try:
                    policy.reset()
                    pred = policy.predict_action_chunk(b).float()
                    tgt = b["action"].float()
                    h = min(pred.shape[1], tgt.shape[1])
                    m = tgt.new_ones(tgt.shape[:2])
                    pad = b.get("action_is_pad")
                    if pad is not None:
                        m = (~pad.bool()).to(tgt.dtype)
                    m = m[:, :h].unsqueeze(-1)
                    d = tgt.shape[-1] - 1        # drop the gripper channel
                    amp_pred = float((pred[:, :h, :d].abs() * m).sum()
                                     / m.sum().clamp(min=1) / d)
                    amp_tgt = float((tgt[:, :h, :d].abs() * m).sum()
                                    / m.sum().clamp(min=1) / d)
                except Exception:
                    amp_pred = amp_tgt = 0.0
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if was_training:
            policy.train()
        return tot / max(1, n), amp_pred, amp_tgt

    # Training loop
    print("Starting training loop...")
    done = False
    prog_bar = tqdm(total=training_steps, desc="Training Progress", initial=step)
    while not done:
        epoch += 1
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Enrich batch with task description strings
            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                batch["task_description"] = batch["task"]
            elif task_idx_to_description and "task_index" in batch:
                task_indices = batch["task_index"]
                if isinstance(task_indices, torch.Tensor) and task_indices.dim() > 1:
                    task_indices = task_indices[:, 0]
                batch["task_description"] = [task_idx_to_description.get(int(ti), "") for ti in task_indices]

            # Apply instruction rewriting if enabled (for LIBERO spatial grounding)
            if rewrite_instructions and "task_description" in batch:
                batch["task_description"] = [
                    rewrite_instruction(t, random_augment=rewrite_augment)
                    for t in batch["task_description"]
                ]

            batch = apply_image_augmentations(batch, camera_keys, image_transforms)
            if awr is not None:
                if "episode_index" not in batch:
                    raise KeyError(
                        "--awr_rewards needs episode_index in the batch to join "
                        "the weights, and this dataset does not provide it.")
                _di = batch.get("dataset_index")
                batch["awr_weight"] = awr.lookup(
                    batch["episode_index"].reshape(-1).tolist(),
                    None if _di is None else _di.reshape(-1).tolist())
            batch = apply_joint_augmentations(
                batch, abs_sigma=state_noise_abs,
                frac_sigma=state_noise_frac,
                state_std=state_std_t, prob=state_noise_prob)

            if step == 0:
                raw_st = batch["observation.state"].float()
                print(f"\nRaw (pre-norm) observation.state: min={raw_st.min():.4f}  max={raw_st.max():.4f}  std={raw_st.std():.4f}")

            batch = preprocessor(batch)

            if step == 0:
                pad_key = next((k for k in ("action_is_pad", "actions_id_pad") if k in batch), None)
                if pad_key is None:
                    print("WARNING: no action pad key found in batch — padded episode steps will pollute loss!")
                    print(f"  Available keys: {[k for k in batch.keys() if 'pad' in k.lower() or 'action' in k.lower()]}")
                else:
                    pad_frac = batch[pad_key].float().mean().item()
                    print(f"Action pad key='{pad_key}', pad fraction in first batch: {pad_frac:.2%}")

            # Forward & Backward
            # Arm the attention-mass diagnostic on the same cadence as
            # gradient analysis. The model self-disarms after one capture.
            if step % progress_update_freq == 0:
                policy.model._capture_attention_stats = True

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                loss, _ = policy.forward(batch)

            if loss.item() > 100 and step < 2000:
                act = batch["action"].float()
                st = batch["observation.state"].float()
                print(f"\n[DIAG step={step}] loss={loss.item():.1f}")
                print(f"  action  : min={act.min():.2f}  max={act.max():.2f}  std={act.std():.3f}")
                print(f"  state   : min={st.min():.2f}  max={st.max():.2f}  std={st.std():.3f}")
                pad_key = next((k for k in ("action_is_pad", "actions_id_pad") if k in batch), None)
                if pad_key is not None:
                    print(f"  pad frac: {batch[pad_key].float().mean().item():.2%}")

            loss.backward()

            if step % progress_update_freq == 0:
                _log_gradient_analysis(policy, step)

            trainable_params = [p for p in policy.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if step % progress_update_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                prog_bar.set_description(f"Epoch {epoch}, Step {step}")
                prog_bar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "lr": f"{lr:.2e}",
                    "grad_norm": f"{grad_norm:.2f}"
                })
                # tqdm writes to STDERR and redraws in place with \r. Under any
                # redirection that keeps only stdout -- or in a log file, where
                # the carriage returns collapse into one unreadable line -- the
                # loss simply disappears, while the gradient analysis (plain
                # print, stdout) survives. Emit it to stdout too, on the same
                # cadence, so the number does not depend on how the run was
                # launched. `total` is the full objective; the analysis block's
                # `main` above it is the flow term alone.
                print(f"  step {step}  total {loss.item():.4f}  lr {lr:.2e}  "
                      f"grad_norm {grad_norm:.2f}", flush=True)

            if val_loader is not None and step > 0 and step % val_every == 0:
                v, ap, at = run_eval_loss(val_loader)
                f, _, _ = run_eval_loss(fit_loader)
                gap = 100.0 * (v - f) / max(abs(f), 1e-9)
                ratio = ap / at if at > 0 else float("nan")
                print(f"\n  VAL @ {step}   "
                      f"fit(train eps) {f:.4f}   held-out {v:.4f}   "
                      f"gap {gap:+.1f}%"
                      f"   [both eval mode, no augmentation]\n"
                      f"      |action| predicted {ap:.4f} vs target {at:.4f}"
                      f"   ratio {ratio:.3f}"
                      f"   (>1 = the policy is taking BIGGER steps than the demos)")
                with open(output_directory / "val_log.jsonl", "a") as fh:
                    fh.write(json.dumps({"step": step, "fit": f, "heldout": v,
                                         "gap_pct": gap, "amp_pred": ap,
                                         "amp_target": at, "amp_ratio": ratio}) + "\n")

            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                policy.config.training_step = step
                policy.config.training_epoch = epoch
                policy.config.optimizer_lr = optimizer.param_groups[0]["lr"]
                policy.config.current_lr = optimizer.param_groups[0]["lr"]
                policy.config.training_steps_total = training_steps
                policy.save_pretrained(checkpoint_dir)
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pth")
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                print(f"\nCheckpoint saved at step {step}")

            step += 1
            if step % progress_update_freq == 0 or step >= training_steps:
                prog_bar.update(progress_update_freq)
                prog_bar.set_description(f"Epoch {epoch}, Step {step}")

            if step >= training_steps:
                done = True
                prog_bar.close()
                break
    prog_bar.close()

    # Final save
    policy.config.training_step = step
    policy.config.training_epoch = epoch
    policy.config.optimizer_lr = optimizer.param_groups[0]["lr"]
    policy.config.current_lr = optimizer.param_groups[0]["lr"]
    policy.config.training_steps_total = training_steps
    policy.save_pretrained(output_directory)
    torch.save(optimizer.state_dict(), output_directory / "optimizer_state.pth")
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, nargs="+", default=["lerobot/libero"],
                        help="One or more LeRobot dataset ids. Multiple are concatenated and "
                             "must share a homogeneous schema (same robot/cameras/dims/fps); "
                             "their normalization stats are aggregated.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute DiT activations in backward to save memory.")
    parser.add_argument("--max_episode_index", type=int, default=None,
                        help="Filter to episodes with index <= this value "
                             "(piper_arm holdout convention; omit for full dataset).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="DataLoader batch size (default: 64).")
    parser.add_argument("--n_obs_steps", type=int, default=None,
                        help="Frames of observation.state FETCHED per sample "
                             "(default: 2). Note that wilro then keeps only the "
                             "last one: wilro_model slices state_tok[:, -1:], so "
                             "anything above 1 is fetched, encoded and thrown "
                             "away. It changes what the dataloader carries, not "
                             "what the model sees -- 1 is strictly cheaper and "
                             "numerically identical. The model has NO velocity "
                             "input either way.")
    parser.add_argument("--val_episodes", type=int, default=0,
                        help="Hold out this many EPISODES for validation, "
                             "allocated proportionally across --dataset_id so a "
                             "mixed run does not measure only the larger set. "
                             "0 disables. Splitting by frame instead would put "
                             "neighbouring frames of one episode on both sides "
                             "and report a gap near zero however badly the "
                             "model memorised.")
    parser.add_argument("--load_image_size", type=int, default=0,
                        help="Resize frames to NxN in the DATALOADER (0 = keep "
                             "native, the default). The model pads to square "
                             "and interpolates to vision_input_size anyway, so "
                             "this costs nothing there and cuts what every "
                             "worker must decode, hold, pin and copy by "
                             "(N/native)^2 -- 480 -> 384 is 0.64x. Opt-in "
                             "because v2.Resize antialiases and the model's "
                             "interpolate does not, so a run using it is not "
                             "bit-comparable to one without.")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Batches each worker keeps queued (default: 2). "
                             "Total decoded frames held is batch_size x "
                             "num_workers x this; 1 halves it.")
    parser.add_argument("--download_progress", action="store_true",
                        help="Keep huggingface_hub's per-file progress bars. "
                             "Off by default: at ~14k files they redraw "
                             "thousands of lines and bury the schema and "
                             "normalisation output printed before training. "
                             "Turn on for a genuinely first-time pull.")
    parser.add_argument("--cache_sync", action="store_true",
                        help="Re-verify every file of every dataset against the "
                             "hub at launch (the old always-on behaviour). "
                             "~14k requests for a converted VLABench before the "
                             "first step; only needed when the remote may have "
                             "changed under an existing cache.")
    parser.add_argument("--lora_rank", type=int, default=None,
                        help="LoRA rank on the SigLIP ViT (default: 16). The "
                             "vision adapter is ~0.1%% of this model's trainable "
                             "parameters -- the DiT is 99.3%% -- so this is not "
                             "a lever on OVERALL capacity, and the model already "
                             "overfits. It is a lever on where adaptation is "
                             "ALLOWED: grounding lives in the encoder, and the "
                             "encoder is where almost nothing trains.")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA alpha (default: 2 x rank). LoRALinear scales "
                             "by alpha/rank, so leaving alpha fixed while "
                             "raising rank halves the adapter's strength. The "
                             "default tracks rank to keep that ratio at the "
                             "shipped 32/16 = 2.0.")
    parser.add_argument("--vision_lora_num_layers", type=int, default=None,
                        help="How many trailing SigLIP ViT layers get adapters "
                             "(default: 8; SmolVLM2-500M's ViT has 27). Text "
                             "LoRA stays at 0 and should: the encoder-decoder "
                             "detaches the VLM KV cache, so no gradient reaches "
                             "the text tower to train an adapter with.")
    parser.add_argument("--vision_token_source", choices=("vlm", "resnet"),
                        default="vlm",
                        help="Where Robot CA's K/V come from. 'vlm' "
                             "(default, what ships) reads a SigLIP ViT layer with a "
                             "frozen base -- about 0.39M trainable in the "
                             "robot-visual path. 'resnet' restores the separate "
                             "trainable ResNet-18 (3.0M measured, NOT the 11.7M of a "
                             "stock ResNet-18 -- layer4 is 72 percent of that and is "
                             "excluded) that this model used until "
                             "2026-07-06 and that wiltechs_moe still uses; removing "
                             "it there cost 34 points of spatial success. It "
                             "REPLACES the VLM source rather than running beside it "
                             "-- a parallel second encoder has been measured getting "
                             "gated off. Compare against sft-40k (68.2), and hold "
                             "--lora_rank/--vision_lora_num_layers at 16/8 when you "
                             "do, or the two capacity changes are not separable.")
    parser.add_argument("--num_experts", type=int, default=4,
                        help="Independent expert decoders, each cross-attending "
                             "to its OWN disjoint band of VLM layers. "
                             "num_experts x expert_num_layers must not exceed "
                             "the VLM's 32 text layers.")
    parser.add_argument("--expert_num_layers", type=int, default=8,
                        help="Layers per expert (default 8). wiltechs_moe uses 9 "
                             "on Qwen3-VL's 36 layers; SmolVLM2 has 32, so 4 x 8 "
                             "is the exact fit here and 4 x 9 raises.")
    parser.add_argument("--dit_hidden_size", type=int, default=960,
                        help="Expert width. Only 960 (== the VLM hidden size) is "
                             "supported: at that width the experts' self- and "
                             "cross-attention share the VLM's 15/5/64 geometry "
                             "and reuse wilro's DiTLayer unchanged.")
    parser.add_argument("--vlm_capture_layers", type=str, default="",
                        help="Comma-separated VLM layer indices to capture. "
                             "Empty = all 32, which is what 4 x 8 wants.")
    parser.add_argument("--resnet_expert_adapter_dim", type=int, default=0,
                        help="Give each expert its own bottleneck MLP over the "
                             "SHARED vision tokens (0 = off). The tokens are one "
                             "set read by every expert, so their encoder gets a "
                             "router-weighted sum of four demands and has to "
                             "compromise; this lets the trunk stay generic while "
                             "each expert learns its own read, at ~0.5M per "
                             "expert at dim 256. Two caveats: it is partly "
                             "redundant with each expert's own value projection, "
                             "which is already a per-expert linear read of these "
                             "tokens; and it makes router collapse MORE "
                             "expensive, since a starved expert's adapter gets "
                             "no gradient at all (mitigated by the zero-init "
                             "residual, so untrained means identity). Default off "
                             "because wiltechs_moe scores 92 WITHOUT it -- turning "
                             "it on for the first run makes that comparison "
                             "unattributable.")
    parser.add_argument("--router_temperature", type=float, default=1.0,
                        help="Softmax temperature on the router logits.")
    parser.add_argument("--router_top_k", type=int, default=0,
                        help="0 = soft mixture over every expert. >0 keeps only "
                             "the top-k, which makes the forward cheaper but "
                             "removes the gradient that keeps unused experts alive.")
    parser.add_argument("--router_balance_weight", type=float, default=0.1,
                        help="Weight on CV^2 of expert usage. Collapse to a "
                             "single expert is the known failure mode of this "
                             "architecture; the router also injects fixed "
                             "N(0, 0.5) logit noise during training for the same "
                             "reason. Read BOTH the usage line and the "
                             "per-sample max_w below it -- the batch mean can "
                             "look uniform while every sample is collapsed.")
    parser.add_argument("--resnet_tokens", type=int, default=64,
                        help="ResNet source only: pooled tokens per camera "
                             "(perfect square for avg pooling). 64 at "
                             "--resnet_input_size 256 gives 32 px/token, "
                             "parity with the VLM's merged patches. moe's historical "
                             "16 gives 64 px/token, i.e. half the granularity of the "
                             "frozen backbone it is supposed to sharpen. Cost is per "
                             "DiT layer and per camera.")
    parser.add_argument("--awr_rewards", type=str, nargs="+", default=[],
                        help="Path to awr_rewards.json written by "
                             "train_rft.py --rft.collect_only --rft.keep_failures. "
                             "Turns this into ADVANTAGE-WEIGHTED REGRESSION: the "
                             "same flow-matching loss with a per-sample weight "
                             "exp(A/beta). No importance ratio, so the corpus can "
                             "be trained on for many epochs, and the gradient "
                             "keeps the character of supervised learning instead "
                             "of a noise-dominated policy gradient. Empty = off.")
    parser.add_argument("--awr_dataset_index", type=int, nargs="+", default=[0],
                        help="Which --dataset_id the awr_rewards sidecar "
                             "describes (0 = the first). Every other dataset in "
                             "the mix gets weight 1.0, which is what you want "
                             "for demos: they are the reference behaviour, not "
                             "something to reweight. Mixing demos in is the "
                             "defence against the corpus being one suite wide.")
    parser.add_argument("--awr_group_by", default="task",
                        choices=("task", "state"),
                        help="What the advantage is standardised within. "
                             "'task' (default) makes beta mean the same thing "
                             "across tasks. 'state' standardises within (task, "
                             "init_state), which cancels LAYOUT DIFFICULTY the "
                             "way GRPO's group baseline does -- per task, a "
                             "200-step success may only mean the object started "
                             "far away, and `steps` cannot tell that from a "
                             "fumble. It needs several rollouts per state to "
                             "have anything to compare: the collector's default "
                             "sweep gives ONE, so every group is degenerate. At "
                             "libero_10's 71%% it takes 5 per state to get the "
                             "degenerate share under 20%%, i.e. 2500 episodes. "
                             "Use it with --rft.iterations 5, or stay on 'task' "
                             "and use --awr_reward success, which does not read "
                             "length at all.")
    parser.add_argument("--awr_beta", type=float, default=1.0,
                        help="AWR temperature. Large => all weights 1, i.e. plain "
                             "BC on everything including failures, no improvement. "
                             "Small => the weight collapses onto the single best "
                             "episode, maximum greed and overfit. beta 1 on this "
                             "policy's staged spread gives roughly 55:1 best-to-worst.")
    parser.add_argument("--awr_clip", type=float, default=20.0,
                        help="Cap on exp(A/beta). Without it one lucky episode "
                             "dominates the batch.")
    parser.add_argument("--awr_reward", type=str, default="success",
                        choices=["success", "fast_success"],
                        help="What the advantage is computed from. 'fast_success' "
                             "discounts by episode length: a slow success fumbled "
                             "and recovered, a fast one did not, and on this "
                             "policy successes average 80 chunks against 259 for "
                             "failures.")
    parser.add_argument("--awr_drop_failures", action="store_true",
                        help="Zero the weight of every failed episode instead "
                             "of leaving it at a low one, i.e. train only on "
                             "the successes (graded among themselves). The "
                             "standardised weight alone does NOT make failures "
                             "harmless: on the goal corpus they are the "
                             "episodes that never closed the gripper -- 31.2%% "
                             "of rollout frames command a close against 47.5%% "
                             "of demo frames -- and at weight ~0.7 they taught "
                             "the policy to hover. libero_goal T0, a "
                             "grasp-and-pull, went 90%% -> 35%% after 4000 "
                             "steps. Weights are still renormalised to mean 1, "
                             "so the corpus keeps the same total pull; it just "
                             "lands on the successes.")
    parser.add_argument("--awr_min_weight", type=float, default=0.0,
                        help="Floor on the weight before renormalisation, "
                             "applied BEFORE --awr_drop_failures so failures "
                             "still go to zero. Use it when a small beta has "
                             "starved the weaker successes and the corpus has "
                             "effectively collapsed onto its best few "
                             "episodes. 0 disables.")
    parser.add_argument("--state_noise_abs", type=float, default=0.01,
                        help="Gaussian sigma added to observation.state in RAW "
                             "units (metres / radians), the historical default "
                             "and what every result in the tracker was trained "
                             "with. One number across dims whose natural scales "
                             "differ by 65x: on LIBERO it lands as 2.6-9.5%% of "
                             "std on position, 1.1-3.1%% on rotation, and 70%% on "
                             "the gripper finger joints. Pass 0 to disable.")
    parser.add_argument("--state_noise_frac", type=float, default=0.0,
                        help="Scale the state noise PER DIM instead: sigma_i = "
                             "frac * std_i, so the knob means one thing "
                             "everywhere. 0 (default) keeps --state_noise_abs. "
                             "frac 0.04 is close to the historical strength on "
                             "position and rotation while taking the gripper "
                             "channel from 70%% down to 4%% -- a near-null change "
                             "on six of eight dims and a real one on two, which "
                             "is why it is opt-in rather than the default.")
    parser.add_argument("--state_noise_prob", type=float, default=0.5,
                        help="Probability per batch that state noise is applied "
                             "at all (default 0.5, the historical value).")
    parser.add_argument("--resnet_fine_cameras", type=str, nargs="+", default=None,
                        help="Cameras that get a DENSER ResNet grid than "
                             "--resnet_tokens, e.g. the wrist view, which carries "
                             "contact geometry while the third-person view only "
                             "supplies coarse approach context. Same backbone, "
                             "different pooling, NO extra parameters. Must be "
                             "given with --resnet_fine_tokens. Names are checked "
                             "against the dataset: a typo would otherwise fall "
                             "back to --resnet_tokens silently.")
    parser.add_argument("--resnet_fine_tokens", type=int, default=0,
                        help="Token grid for --resnet_fine_cameras (perfect "
                             "square). The native map is (input_size/16)^2 -- 196 "
                             "at 224px -- so 196 there is a 1:1 read and anything "
                             "lower throws spatial resolution away. Raises the DiT "
                             "sequence length, which is quadratic in attention "
                             "FLOPs; lower --resnet_tokens for the other cameras "
                             "before lowering --batch_size, because a batch change "
                             "breaks the step-to-samples mapping against earlier "
                             "runs.")
    parser.add_argument("--vision_input_size", type=int, default=384,
                        help="Resolution the FROZEN SmolVLM2 tower reads. Must "
                             "be divisible by 64 (patch 16 x pixel-shuffle 4). "
                             "Tokens per camera = (v/64)^2: 36 at the 384 "
                             "default, 64 at 512. **512 is the tower's NATIVE "
                             "pretrained resolution -- 384 runs it BELOW what it "
                             "was trained at**, so this is not a safe default, "
                             "it is a cost saving. Raising it costs ~1.78x the "
                             "vision patches and lengthens the KV the experts "
                             "cross-attend to; budget ~1.4x step time. Changing "
                             "it on a resume is legal but needs LR left to "
                             "re-adapt -- see the warning the preflight prints.")
    parser.add_argument("--resnet_input_size", type=int, default=256,
                        help="ResNet input resolution. 256 is the native LIBERO "
                             "frame, so no resample happens.")
    parser.add_argument("--resnet_pool", choices=("avg", "attn"), default="avg",
                        help="'avg' adaptive average pooling (what moe runs at 92). "
                             "'attn' is AttentionPool2d with grid-seeded learned "
                             "queries; its query count is fixed at construction, so "
                             "it cannot serve the motion path's second grid.")
    parser.add_argument("--use_state_history", action="store_true",
                        help="Stop slicing the state window to its last frame, so "
                             "--n_obs_steps finally changes what the model sees "
                             "(today it does not: the extra frames are encoded and "
                             "discarded). The leak control is already run -- the "
                             "momentum shortcut sits 33x above the model's own "
                             "residual -- and a four-condition dose-response says "
                             "the channel carries real information. Counter-risk: on "
                             "the sibling's task 5 three independent corruptions of "
                             "the window each cut time-to-success 195 to ~110 steps, "
                             "and wilro pins its step cap on exactly the five tasks "
                             "with that signature. Read the result per task.")
    parser.add_argument("--resnet_motion_tokens", type=int, default=0,
                        help="Extra tokens per camera from differencing the ResNet "
                             "FEATURE MAPS of the current and an older frame, "
                             "zero-init gated. 0 disables. Needs "
                             "--vision_token_source resnet. The VLM still sees one "
                             "frame by design. This is the only flag here that "
                             "changes the dataloader: it requests a second camera "
                             "frame, doubling decode bandwidth per camera.")
    parser.add_argument("--resnet_motion_stride", type=int, default=1,
                        help="How many frames back the differenced frame comes "
                             "from. At 10Hz with n_action_steps=2 the policy "
                             "re-plans every 200ms, so 1 frame = 100ms pairs "
                             "naturally.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate (default: the config's 1e-4). "
                             "The cosine is built around this, and the resume "
                             "path rebuilds param_groups from it, so an "
                             "explicit value survives --resume_from_checkpoint "
                             "-- unlike the sibling trainer before 4caed2d. "
                             "Refining an already-trained policy on collected "
                             "rollouts wants ~1e-5; train_rft.py's own default "
                             "is 1e-5.")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Linear warmup steps before the cosine (default: "
                             "1500). Lower it for short refinement runs, where "
                             "1500 can be most of the budget.")
    parser.add_argument("--start_step_override", type=int, default=-1,
                        help="Restart the step counter at this value when "
                             "loading a checkpoint (-1 = keep the checkpoint's, "
                             "the default). Pass 0 to fine-tune on a DIFFERENT "
                             "dataset: without it the checkpoint's step is "
                             "inherited, the cosine is fast-forwarded to it, "
                             "and the run starts mid-decay with no warmup and "
                             "a step budget short by however far the previous "
                             "run got.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes (default: 8). Each "
                             "forks a copy of the dataset; Python refcounting "
                             "then un-shares the copy-on-write pages holding "
                             "the Arrow table, so host RAM climbs for hundreds "
                             "of steps until the runtime SIGINTs the process -- "
                             "which prints a bare ^C and no traceback. On Colab "
                             "with a 575k-row set, 2-4 is the safe range.")
    parser.add_argument("--progress_update_freq", type=int, default=200,
                        help="Steps between the gradient/attention diagnostic "
                             "and the progress-bar refresh (default: 200). The "
                             "attention capture re-runs the last DiT layer's "
                             "softmax at (B, H, L, L) in fp32 under no_grad -- "
                             "transient, but the largest single allocation in "
                             "the step. Raise it to move that cost, which is "
                             "also how to test whether it is what is killing a "
                             "run at a multiple of this number.")
    parser.add_argument("--val_every", type=int, default=500,
                        help="Steps between validation passes (default: 500).")
    parser.add_argument("--val_max_batches", type=int, default=20,
                        help="Batches per validation pass (default: 20).")
    parser.add_argument("--training_steps", type=int, default=None,
                        help="Total optimizer steps (default: 200000). The "
                             "cosine LR schedule spans this, so it is not a "
                             "stop-whenever ceiling -- interrupting a 200k run "
                             "early leaves the LR mid-cosine and the model "
                             "never annealed. On resume an explicit value "
                             "overrides the checkpoint's and rebuilds the "
                             "schedule.")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1,
                        help="Weight for the language-permute contrastive loss "
                             "(default: 0.1). Bump to ~0.5 for LIBERO / datasets "
                             "with diverse task descriptions.")
    parser.add_argument("--contrastive_margin", type=float, default=0.05,
                        help="Hinge margin on MSE between v_t and v_wrong "
                             "(default: 0.05). Bump to ~0.2 to force the model "
                             "to differentiate velocities by language.")
    parser.add_argument("--contrastive_hard_negatives", action="store_true",
                        help="Pair each sample with its hardest in-batch negative (most word "
                             "overlap, different instruction) instead of a random one, so the "
                             "contrastive hinge pressures fine-grained object grounding (the "
                             "confusable minimal pairs that fail at eval) rather than trivially-"
                             "different tasks. Expect the reported contrastive value to spike "
                             "when first enabled, then decline. Off = legacy random pairing.")
    parser.add_argument("--paraphrase_augment", action="store_true",
                        help="Draw a different phrasing of the same instruction "
                             "per sample per step, so the surface string stops "
                             "being a usable key. Measured on the sibling "
                             "(wiltechs-x-114k, libero_spatial T7): 60%% on its "
                             "own instruction, 0%% on a PARAPHRASE of that same "
                             "instruction -- it had memorised the ~40 strings "
                             "and was retrieving, not reading. Table in "
                             "src/libero_paraphrase.py; the original string is "
                             "always among the variants, since eval uses it.")
    parser.add_argument("--paraphrase_limit", type=int, default=8,
                        help="Cap on variants per instruction (0 = all).")
    parser.add_argument("--paraphrase_file", default="",
                        help="JSON table overriding the built-in one, for "
                             "instructions it does not cover. Draft it with "
                             "python -m libero_paraphrase --dataset_id <id> "
                             "--out f.json, then hand-edit -- templates never "
                             "reach the model unread.")
    parser.add_argument("--paraphrase_min_variants", type=int, default=5,
                        help="Refuse to start when any instruction has fewer "
                             "variants than this. Partial augmentation is worse "
                             "than none: the unvaried tasks keep surface form as "
                             "a key and the run answers nothing.")
    parser.add_argument("--lock_joint_index", type=int, default=None,
                        help="Force one action dim to loss weight 0. Default is "
                             "None: dims are locked FROM THE DATA (std <= 0.1%% "
                             "of the widest), which catches piper_arm's dead "
                             "joint 4 and leaves LIBERO's dim 3 alone. A zero "
                             "weight does not suppress a dim, it makes the model "
                             "sample it from its marginal -- only correct for a "
                             "mechanically locked joint. Pass -1 to force none.")
    parser.add_argument("--kv_capture_strategy", type=str, default="last",
                        choices=["last", "stride2", "custom"],
                        help="Which VLM layers the DiT sources KV from. "
                             "'last' = trailing N layers (most refined, no "
                             "multi-scale). 'stride2' = every other layer, "
                             "end-anchored (multi-scale: shallow DiT reads "
                             "shallow VLM). 'custom' = exactly the layers given "
                             "in --kv_capture_layers (DiT depth = #layers). "
                             "NOT resume-compatible across values.")
    parser.add_argument("--kv_capture_layers", type=str, default="",
                        help="Comma-separated 0-based VLM layer indices for "
                             "--kv_capture_strategy custom, e.g. '3,7,11,15,19,"
                             "23,27,31'. Ignored for last/stride2.")
    parser.add_argument("--cameras", type=str, nargs="+", default=None,
                        help="Subset of cameras to use from the dataset. If not specified, "
                             "all available cameras are used. Example: "
                             "--cameras observation.images.chest observation.images.left_hand")
    parser.add_argument("--noise_temporal_correlation", type=float, default=0.0,
                        help="AR(1) coefficient correlating the flow-matching source "
                             "noise along the action horizon (0=white noise; ~0.9=temporally "
                             "smooth). Source dist changes, so this is NOT inference-only — "
                             "resume from a rho=0 checkpoint and fine-tune to adapt. Too high "
                             "(>0.95) over-smooths sharp/contact motions.")
    parser.add_argument("--video_backend", default=None,
                        choices=("torchcodec", "pyav", "video_reader"),
                        help="Frame decoder. Default is lerobot's own choice, "
                             "which prefers torchcodec where it is available. "
                             "Switch to pyav when a worker dies with "
                             "'Could not push packet to decoder: Invalid data "
                             "found when processing input' -- torchcodec is the "
                             "newer path and is the more likely suspect after an "
                             "environment change. If pyav fails on the same "
                             "sample the file itself is bad and the cache needs "
                             "re-syncing, not the backend switching.")
    parser.add_argument("--n_action_steps_cli", "--n_action_steps", type=int,
                        default=None, dest="n_action_steps_cli",
                        help="TRAINING-SIDE position-weight boundary, not an "
                             "inference setting: compute_loss sets "
                             "pos_w[n_action_steps:] = future_steps_weight. The "
                             "historical 64 equals the horizon, so that slice is "
                             "EMPTY and future_steps_weight has never been in "
                             "effect on any run in this repo. Meanwhile the "
                             "2026-09-13 horizon profile put the EXECUTED prefix "
                             "(positions 0-1) at 0.213 of 'predict nothing' "
                             "against 0.153 at positions 16-31 -- the worst part "
                             "of the chunk -- carrying 3.1%% of the gradient "
                             "weight. 8 raises that to 8.1%%. Every eval overrides "
                             "n_action_steps on its own command line, so this "
                             "does not change what inference executes.")
    parser.add_argument("--gripper_transition_window", type=int, default=2,
                        help="Dilate the gripper-transition mask by +/- this "
                             "many chunk positions; the up-weighted span is "
                             "2*win+1. Measured 2026-09-18 on 8x4-22k-obs2 "
                             "(analyze_gripper_window.py): the residual is "
                             "elevated over the far-field baseline out to d=8-16, "
                             "not d=2 -- 1.99x at d=1, 1.66x at d=2, 1.63x at "
                             "d=3-4, 1.42x at d=5-8, 1.23x at d=9-16. The default "
                             "2 cuts between d=2 and d=3-4, which are equally "
                             "elevated. **4 is what that profile supports.** Note "
                             "the peak is at d=1, NOT at the transition itself "
                             "(d=0 reads 1.49x): the flip is a saturated binary "
                             "value and easy, its neighbours are where the "
                             "positioning has to be right.")
    parser.add_argument("--gripper_transition_thresh", type=float, default=0.5,
                        help="|delta gripper| above which a chunk position counts "
                             "as a transition, in NORMALISED units. On "
                             "lerobot/libero this is not a tunable: the gripper "
                             "is strictly binary, |dg| is 0.000 at every "
                             "percentile up to p95 and exactly 2.002 at p99, so "
                             "anything in (0, 2) gives identical masks. It "
                             "matters only for a dataset whose gripper RAMPS -- "
                             "the preflight below reports which case you are in.")
    parser.add_argument("--gripper_phase_weight", type=float, default=1.0,
                        help="Up-weight the flow-matching loss on frames near a gripper "
                             "open<->close transition (grasp/release) — the precision-critical "
                             "moments uniform MSE dilutes. 1.0=off (default); try 2-4 to sharpen "
                             "placement. Gripper assumed to be the last action dim (LIBERO OSC).")
    parser.add_argument("--time_sampling", type=str, default="uniform",
                        choices=["uniform", "lognormal"],
                        help="Flow-matching timestep sampling. 'lognormal' (SD3 logit-normal) "
                             "biases toward low t (x_t≈actions), spending more capacity on the "
                             "fine-detail denoising that sets placement precision.")
    parser.add_argument("--time_lognormal_mean", type=float, default=-0.5,
                        help="Mean of the logit-normal (only if --time_sampling lognormal). "
                             "More negative => more mass at low t (finer detail).")
    parser.add_argument("--time_lognormal_std", type=float, default=1.0,
                        help="Std of the logit-normal (only if --time_sampling lognormal).")
    parser.add_argument("--rewrite_instructions", action="store_true",
                        help="Apply instruction rewriting from task_rewrites.py for LIBERO "
                             "spatial grounding (e.g., ramekin -> visual description, "
                             "'between' -> 'closer to'). Rewritten instructions are used "
                             "for both VLM encoding and contrastive hard negatives.")
    parser.add_argument("--rewrite_augment", action="store_true",
                        help="When --rewrite_instructions is enabled, randomly choose between "
                             "original and rewritten instruction (50/50) for each sample. "
                             "This trains the model to understand BOTH phrasings.")
    args = parser.parse_args()
    # -1 must NOT be folded to None any more: None now means "decide from the
    # data" and -1 means "lock nothing at all". Folding them together would
    # make --lock_joint_index -1 silently re-enable auto-detection, which on
    # piper_arm still locks joint 4. The block in train() reads -1 as
    # out-of-range and produces an empty lock list, which is the intent.
    # Parse the comma-separated custom layer list into ints.
    args.kv_capture_layers = [
        int(tok) for tok in args.kv_capture_layers.split(",") if tok.strip() != ""
    ]
    if args.kv_capture_strategy == "custom" and not args.kv_capture_layers:
        parser.error("--kv_capture_strategy custom requires --kv_capture_layers")
    train(**vars(args))
