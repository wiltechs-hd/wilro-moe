# WILRO-MoE

A small vision-language-action policy for LIBERO, and the inference-time noise
search that lifts it with the weights frozen.

- **Policy.** A frozen SmolVLM2-500M encoder feeds a mixture-of-experts
  flow-matching action head (8 experts × 4 DiT layers, each expert
  cross-attending into its own band of the VLM's KV cache), plus ResNet-18
  tokens from the agent and wrist cameras and a two-frame state history.
- **Golden tickets.** Replace the flow-matching initial noise `x_1 ~ N(0, I)`
  with one searched constant vector per task ([Patil et al., 2026](https://arxiv.org/abs/2603.15757)).
- **Noise cycles.** On long-horizon tasks one frozen vector collapses success,
  because a ~76-chunk episode reuses the same draw 76 times. Cycling `m` fixed
  vectors (chunk `k` uses `t[k mod m]`) keeps the rollout deterministic and
  makes the search work there.

## Results

LIBERO, canonical initial states 0–19, 20 episodes per task, frozen weights,
checkpoint [`ISdept/wilro-wilromoe-8x4-22k-obs2`](https://huggingface.co/ISdept/wilro-wilromoe-8x4-22k-obs2).

| Suite | Per-chunk redraw | With tickets | Ticket type |
|---|---:|---:|---|
| Object | 86.5 | 96.0 | single vector |
| Spatial | 88.5 | 97.5 | single vector |
| Goal | 88.0 | 94.5 | single vector |
| LIBERO-Long | 71.0 | 81.0 | searched 4-cycle |
| **Average** | **83.5** | **92.3** | |

On LIBERO-Long with every searched ticket active the suite reads 79.0
(32 discordant initial states won to 16, two-sided sign test p = 0.029); 81.0
lets the one task whose ticket loses to the redraw fall back to it. An
unsearched random 4-cycle scores 63.5: cycling alone does not help, it is what
makes the search possible.

## Install

Python 3.10. The numbers above were measured with:

```bash
pip install -r requirements.txt
```

LIBERO itself (`libero`, `robosuite`, `mujoco`) follows LeRobot's LIBERO
setup. The versions every reported result used are pinned in
`requirements.txt`; `eval_libero.py` writes the exact environment into every
result JSON (`env.digest`) so two runs can be checked for comparability.

All commands run from `src/`.

## Evaluate

```bash
python eval_libero.py \
  --checkpoint ISdept/wilro-wilromoe-8x4-22k-obs2 \
  --suites libero_10 --episodes 20 --n_action_steps 2 \
  --out long.json
```

With tickets, add `--noise_tickets <dir>/golden_tickets.safetensors`. A single
vector or cycle can be tried with `--noise_ticket vec.npy` or
`--noise_cycle cycle.npy` (shape `(m, 64, 7)`).

Three settings are load-bearing and easy to get wrong:

- **`--n_action_steps 2`.** The checkpoint ships 64, under which the policy
  replans a handful of times per episode and scores near zero. The eval warns.
- **10 Hz control.** The LIBERO dataset is 10 Hz; LeRobot's stock env is 20 Hz.
  `eval_libero.py` defaults to 10.
- **Canonical layouts.** Stock LeRobot resets LIBERO in an order that discards
  the canonical initial state and serves placement-sampler layouts about ten
  times wider. `libero_env_fixed.patch_lerobot_libero()` restores LIBERO's own
  order; every script here calls it. `python libero_env_fixed.py` checks it.

## Search tickets

```bash
python search_golden_ticket.py \
  --checkpoint ISdept/wilro-wilromoe-8x4-22k-obs2 \
  --suites libero_10 --task_ids 0 \
  --cycle 4 --tickets 128 --envs_per_tier 5 --tiers 3 \
  --certify_layouts 15 --out ./long_tickets
python ticket_bundle.py report ./long_tickets
```

Use `--cycle 1` for single vectors (short suites). The search is random search
with sequential halving over tiers of initial states 20–34, with floors taken
from the unmodified policy on exactly the layouts each candidate ran, and an
optional certification batch on states 35–49. Initial states 0–19 are never
read by the search tiers. It resumes from its progress file after a crash.

`ticket_bundle.py` manages the result: `report`, `disable|enable` (falls a task
back to the redraw without deleting the ticket), `put`, `revert`, and `merge`
(combine bundles searched on different machines). `try_runners.py` evaluates
the runner-up candidates of a finished search. `plot_tickets.py` draws a
selected ticket against its candidate pool.

## Train

The released checkpoint was trained on `lerobot/libero` (all four suites) in
two stages: an 8×4 run from scratch at 224 px, then a continuation that added
the two-frame state history and 256 px ResNet input. The second stage:

```bash
python train_wilro_moe.py \
  --output_dir ./outputs/wilro_moe_8x4_res \
  --resume_from_checkpoint <stage-1 checkpoint> \
  --dataset_id lerobot/libero \
  --vision_token_source resnet --resnet_input_size 256 --resnet_tokens 144 \
  --resnet_fine_cameras observation.images.image2 --resnet_fine_tokens 256 \
  --n_obs_steps 2 --use_state_history \
  --gripper_phase_weight 3.0 --vision_lora_num_layers 0 \
  --num_experts 8 --expert_num_layers 4 \
  --batch_size 48 --router_balance_weight 0.1 \
  --paraphrase_augment --gradient_checkpointing \
  --lr 1e-4 --warmup_steps 1500 --state_noise_abs 0 \
  --training_steps 25000
```

## Layout

```
src/
  models/wilro_moe/      the policy (config, model, LeRobot wrapper, processors)
    ARCHITECTURE.md      design notes
  eval_libero.py         LIBERO evaluation
  search_golden_ticket.py, ticket_bundle.py, try_runners.py, plot_tickets.py
  train_wilro_moe.py
  libero_env_fixed.py    canonical-layout reset order
  test_search_resume.py  GPU-free test that a killed search resumes identically
```

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot),
[SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). The ticket
intervention is from Patil et al., *You've Got a Golden Ticket: Improving
Generative Robot Policies With A Single Noise Vector* (2026).

## License

Apache-2.0. See [LICENSE](LICENSE).
