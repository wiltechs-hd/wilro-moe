"""Search for a golden ticket: one constant x_1 that beats sampling from N(0,I).

Patil et al. 2026, "You've Got a Golden Ticket". A frozen diffusion/flow policy
is improved by replacing the per-step Gaussian draw with a single well-chosen
noise vector, found by Monte-Carlo rollout search. No weights change, no new
network is trained.

WHY THIS FITS THIS PROJECT'S OWN MEASUREMENTS. The 2026-09-23 best-of-N result
showed chunk-level selection buys nothing here (p=1.0000 against a random-pick
control): four draws at one state are interchangeable. But a `policy_seed`
change -- which is a change to the WHOLE episode's noise -- flips outcomes, and
this file prices the per-chunk re-draw at 25 points. A constant ticket acts at
the episode level, which is the level that was measured to matter.

SEARCH AND EVAL LAYOUTS ARE DISJOINT, BY CONSTRUCTION. LIBERO ships 50
canonical initial states and eval_task maps episode index -> layout id
directly, so a standard 20-episode eval uses ids 0-19 and nothing else. This
searches at --init_state_offset 20 by default, leaving 0-19 untouched: the
ticket is then REPORTED on exactly the layouts every number in the tracker was
measured on, without having been fitted to them.

    python src/search_golden_ticket.py \
        --checkpoint ISdept/wilro-wilromoe-8x4-22k-obs2 \
        --suites libero_goal --task_ids 9 \
        --tickets 128 --out ./tickets

Then, to report it:

    python src/eval_libero.py --checkpoint <same> --suites libero_goal \
        --task_ids 9 --episodes 20 --noise_ticket ./tickets/<file>.npy
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import zlib
from math import comb

import numpy as np
import torch

def binom_ub(k, n, conf=0.90):
    """One-sided Clopper-Pearson upper bound on the true rate given k of n.

    Used to prune candidates that CANNOT plausibly reach the baseline, which
    is the only pruning rule that carries no risk of dropping a good ticket.
    A flat threshold does carry that risk and is also mis-calibrated: 80% sits
    12 points ABOVE goal's 68% search baseline and 6 points BELOW object's
    86.5%, so the same number is too strict on one suite and keeps
    worse-than-Gaussian tickets on the other. At 5 episodes a flat 80% drops a
    truly-75% ticket 37% of the time and a truly-89% ticket 10% of the time --
    and goal T0's winning ticket sits at 0.81-0.89 posterior.
    """
    lo, hi = 0.0, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        cdf = sum(comb(n, i) * m ** i * (1 - m) ** (n - i) for i in range(k + 1))
        lo, hi = (m, hi) if cdf > 1 - conf else (lo, m)
    return (lo + hi) / 2


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Bin(n, p), exact, no scipy."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def dispersion(wins, runs):
    """Do the tickets actually DIFFER, or is the spread just binomial noise?

    This is the question tier 1 exists to answer, and eyeballing the spread
    cannot answer it: with 5 layouts per ticket, 64 IDENTICAL tickets at p=0.7
    still produce observed rates with sd 0.205, i.e. a typical range of
    29%-100%. Any histogram of that looks convincingly "spread out".

    The test is for overdispersion. Under H0 every ticket has the same true
    rate p, so w_i ~ Bin(M, p) and

        chi2 = sum_i (w_i - M p)^2 / (M p (1-p))   ~   chi2(N-1)

    dispersion = chi2/df is 1.0 when the tickets are interchangeable and grows
    with real between-ticket variance. z = (chi2-df)/sqrt(2 df) is the
    normal-approximation score, which is accurate at these df.

    Also returns sd_between, the between-ticket sd left after subtracting the
    binomial part. It is the effect size in success-rate units, and it can sit
    BELOW sd_binomial_only while dispersion is large -- that is not a
    contradiction. When most tickets score zero and a few score high, chi2
    responds to the tail and the variance decomposition does not.

    Fed the CUMULATIVE totals, including tickets already eliminated. The
    halving selects on score, so that looks like it should inflate the
    statistic; simulated under H0 (64 tickets all at p=0.03, three tiers with
    halving) it does not -- cumulative dispersion runs 0.64-0.69, if anything
    conservative -- and cumulative has more episodes behind it than one tier
    does.
    """
    import math
    w = np.asarray(wins, float); r = np.asarray(runs, float)
    keep = r > 0
    w, r = w[keep], r[keep]
    N = len(w)
    if N < 2 or r.sum() == 0:
        return None
    pbar = w.sum() / r.sum()
    if not (0 < pbar < 1):
        return {"n": N, "p_bar": pbar, "note": "every ticket identical "
                "(all 0 or all 1); no variance to test"}
    chi2 = float((((w - r * pbar) ** 2) / (r * pbar * (1 - pbar))).sum())
    df = N - 1
    var_obs = float(np.var(w / r, ddof=1))
    var_bin = float(pbar * (1 - pbar) * np.mean(1.0 / r))
    return {"n": N, "p_bar": round(pbar, 4), "chi2": round(chi2, 1), "df": df,
            "dispersion": round(chi2 / df, 3),
            "z": round((chi2 - df) / math.sqrt(2 * df), 2),
            "sd_between": round(max(0.0, var_obs - var_bin) ** 0.5, 4),
            "sd_binomial_only": round(var_bin ** 0.5, 4)}


import eval_libero as ev
import ticket_bundle as tb


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suites", nargs="+", default=["libero_goal"])
    p.add_argument("--task_ids", nargs="+", type=int, default=None)
    p.add_argument("--out", default="./tickets")
    p.add_argument("--tickets", type=int, default=128,
                   help="Candidates per task. The paper used 1081-1416 per "
                        "LIBERO task; the returns are steep at the start "
                        "because most random tickets are bad, so a few hundred "
                        "already surfaces something.")
    p.add_argument("--envs_per_tier", type=int, default=5,
                   help="Layouts a candidate is scored on in its first tier. "
                        "Survivors accumulate another --envs_per_tier at each "
                        "subsequent tier.")
    p.add_argument("--tiers", type=int, default=3,
                   help="Sequential halving: every candidate is scored on tier "
                        "1, the bottom half is dropped, survivors get a fresh "
                        "disjoint tier, and so on. Cost is about 2 x tickets x "
                        "envs_per_tier instead of tickets x (tiers x "
                        "envs_per_tier), and the deepest survivors are still "
                        "scored at full fidelity. 3 tiers x 5 layouts "
                        "beats 5 x 2 under --prune_mode point: that rule "
                        "compares a RATE to the baseline, and n=2 can only "
                        "express 0, 0.5 and 1, so every task above a 50%% "
                        "baseline collapses to the same 2/2 threshold. n=5 "
                        "separates 5/5, 4/5, 3/5 and 2/5, which is what lets "
                        "one rule serve a 96%% task and a 40%% task at once.")
    p.add_argument("--init_state_offset", type=int, default=20,
                   help="First canonical layout used for SEARCH. 20 keeps the "
                        "reportable 0-19 out of the search entirely. Lower it "
                        "only if you intend to overfit on purpose.")
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--max_episode_steps", type=int, default=0,
                   help="Cap search rollouts shorter than eval's to buy "
                        "candidates: failures run to the cap and dominate the "
                        "wall clock, while successes average ~85 steps. 0 uses "
                        "the env's own cap.")
    p.add_argument("--dataset_id", default=None)
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--n_action_steps", type=int, default=2,
                   help="Steps of each chunk executed before replanning. The "
                        "CHECKPOINT SAYS 64 AND EVERY EVAL IN THIS PROJECT "
                        "PASSES 2 -- leaving it at the checkpoint's value runs "
                        "a policy that replans twice per episode instead of "
                        "150, which is the same mismatch that made the RFT "
                        "collector return 0/200. A ticket is only valid for "
                        "the inference config it was searched under, so this "
                        "must match the eval command.")
    p.add_argument("--vision_input_size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--control_freq", type=int, default=10,
                   help="MUST be 10: the LIBERO datasets are 10 Hz and the "
                        "stock env is 20, so a search at 20 optimises a ticket "
                        "for a policy that is not the one being reported.")
    p.add_argument("--render_gpu", type=int, default=0)
    p.add_argument("--stock_init", action="store_true",
                   help="Use lerobot's unpatched reset order, i.e. the sampler "
                        "distribution. Matches --stock_init in eval and is not "
                        "for anything reportable.")
    p.add_argument("--reported_rate", type=float, default=None,
                   help="This task's success rate in the eval you report, as a "
                        "fraction, e.g. 0.70. The search then checks its own "
                        "Gaussian baseline against it and says so when the two "
                        "diverge, because the search layouts and the reported "
                        "layouts are different scenes and can differ enormously "
                        "in difficulty. goal T9 reads 39%% over layouts 20-34 "
                        "and 70%% over 0-19, so its floor asked candidates to "
                        "clear a 34%% bar while the eval asked them to beat "
                        "70%% -- and its ticket duly beat the search baseline "
                        "and then lost 15 points on the reported set. The three "
                        "tasks whose baselines agreed within 6 points all "
                        "behaved.")
    p.add_argument("--cycle", type=int, default=1,
                   help="Search an m-tuple instead of a single vector; chunk "
                        "k uses element k mod m. m=1 is the paper's golden "
                        "ticket. Higher m exists because a single frozen x_1 "
                        "costs libero_10 T0 fifty points through REPETITION "
                        "rather than through being the wrong vector: a sweep "
                        "of RANDOM cycles read 15%% at m=1, 35%% at 2, and 70%% "
                        "from m=4 on, against Gaussian's 60-73%%, and the "
                        "curve is flat past 4. Four fixed vectors decorrelate "
                        "the noise across chunks while keeping the rollout "
                        "deterministic, which is what training always had and "
                        "a ticket takes away. It also decides whether a task "
                        "is searchable at all: the per-layout pass rate of a "
                        "candidate goes 0.15 -> 0.70, so a 5/5 tier floor is "
                        "cleared once per 13,169 candidates at m=1 and once "
                        "per 6 at m=4.")
    p.add_argument("--require_perfect", action="store_true",
                   help="Floor is 1.0 at every tier, whatever the baseline. "
                        "Use it with --init_state_offset 0 and tiers x "
                        "envs_per_tier = 50 to search EVERY canonical layout: "
                        "a ticket that survives has been verified to solve all "
                        "50, which is the only way to have a deterministic "
                        "100%% rather than an estimate of one. A ticket is one "
                        "fixed vector and a rollout from a given init state is "
                        "then deterministic, so a task has exactly 50 "
                        "distinguishable episodes and no amount of repetition "
                        "buys more evidence -- 30/30 on the held-out 30 leaves "
                        "P(20/20) at 61%%, which is an information ceiling, "
                        "not a method problem. THE REPORTED 20 ARE THEN PART "
                        "OF THE SEARCH; say so. The claim to make is 'this "
                        "ticket solves all 50 canonical layouts', which is "
                        "checkable and true, and not 'it generalises', which "
                        "is untested. Cost is 1/p^50 candidates for a "
                        "per-layout pass rate p: 2 h at p=0.96, 19 h at 0.90, "
                        "225 h at 0.85, out of reach below that. Exact pruning "
                        "makes the pool cheap because a candidate dies at its "
                        "first failure.")
    p.add_argument("--certify_layouts", type=int, default=0,
                   help="After the winner is picked, run IT and Gaussian on "
                        "this many further layouts that took no part in the "
                        "selection, and refuse to bank a ticket that does not "
                        "win there. This is the gate goal T5/T6/T9 needed: all "
                        "three passed a 15-episode check on the search layouts "
                        "and then scored 70/75/55 on the held-out 0-19, "
                        "because the search rate is a MAXIMUM over candidates "
                        "and a maximum is optimistic by construction. "
                        "Certification episodes never enter the selection, so "
                        "the number is unbiased. Cheap: one batch covers "
                        "--num_envs layouts at a time (init_state_stride=1), "
                        "so 15 layouts for the ticket plus 15 for Gaussian is "
                        "4 batches. 15 with the default geometry uses ids "
                        "35-49 and exactly fills the canonical 50.")
    p.add_argument("--certify_min_margin", type=float, default=0.0,
                   help="How far above Gaussian the ticket must land on the "
                        "certification layouts, in rate. 0.0 means a tie is "
                        "enough (which is what determinism wants at the "
                        "ceiling); 0.05 asks for real headroom.")
    p.add_argument("--stratify_layouts", type=int, default=1,
                   help="Deal the search layouts to tiers by measured "
                        "difficulty instead of in id order. Layout difficulty "
                        "varies enormously inside one task -- libero_10 T0 "
                        "Gaussian scored 45/50 on ids 20-24 and 20/50 on "
                        "25-29 -- so in id order the floor a tier enforces is "
                        "whatever its block happened to contain, and tier 1, "
                        "the tier that filters all the candidates, drew a 5/5 "
                        "bar on that task purely by position. Needs the "
                        "baseline to cover tiers x envs_per_tier layouts, "
                        "which the default does. 0 keeps id order.")
    p.add_argument("--allow_downgrade", action="store_true",
                   help="Permit replacing a WORKING ticket in --out. "
                        "save_ticket overwrites unconditionally, so "
                        "--overwrite pointed at the live bundle can retire a "
                        "ticket that is earning episodes and put a worse one "
                        "in its place, with the eval that proved the old one "
                        "already spent. A ticket marked beats_baseline false "
                        "is replaced freely -- eval ignores it, so there is "
                        "nothing to lose. The usual answer is a separate "
                        "--out, which also removes the need for --overwrite.")
    p.add_argument("--retry_failed", action="store_true",
                   help="Re-search tasks whose banked ticket is marked "
                        "beats_baseline false. Off by default because the "
                        "candidates are seeded per task: the same command "
                        "would redraw the same tickets and reach the same "
                        "answer, hours later. Pair it with more --tickets or "
                        "a different --seed, which are the only two things "
                        "that change the draw.")
    p.add_argument("--abandon_on_empty_floor", type=int, default=1,
                   help="Stop a task the moment a tier eliminates every "
                        "candidate. Whatever is carried forward is already "
                        "under the floor and the final check will say NOT "
                        "BETTER than Gaussian, so the remaining tiers only "
                        "confirm it -- an hour, on object T5. 0 keeps going, "
                        "which is only useful for reading the full "
                        "distribution.")
    p.add_argument("--exact_prune", type=int, default=1,
                   help="Inside a tier, drop a candidate the moment its BEST "
                        "remaining outcome falls below the floor: after r of M "
                        "layouts it can finish no higher than (w + M - r) / "
                        "((tier + 1) * M). Arithmetic, not inference, so it "
                        "cannot discard a candidate the tier would have kept. "
                        "It is also what pays for M=5: against a 96%% baseline "
                        "(floor 5/5) one loss is fatal, so tier 1 goes 64 -> "
                        "29 -> 13 -> 6 -> 3 and costs 14 batches instead of "
                        "35. Point mode only; 0 disables.")
    p.add_argument("--baseline_layouts", type=int, default=0,
                   help="Layouts the Gaussian reference is measured on. 0 "
                        "means --envs_per_tier, which is fine at 5 and wrong "
                        "at 2: splitting tier 1 finer to save time would drop "
                        "the baseline to two layouts while the winner is "
                        "judged on ten, and the beats_baseline comparison "
                        "would then be across different layout sets. Set it to "
                        "5 whenever --envs_per_tier is below 5.")
    p.add_argument("--prune_mode", choices=("point", "bound"),
                   default="point",
                   help="How a candidate is compared to its tier's baseline. "
                        "'bound' keeps anything whose --prune_confidence upper "
                        "bound reaches the baseline -- right when the question "
                        "is 'might this ticket be worth using', far too "
                        "lenient when the question is determinism: at n=2 a "
                        "1/2 candidate has a 0.949 bound and survives a 0.96 "
                        "baseline. 'point' keeps only wins/runs >= baseline, "
                        "which self-adapts: 5/5 on a 96%% task, 4/5 on 80%%, "
                        "2/5 on 40%%. libero_10 cannot reach zero failures, so "
                        "a flat 'perfect' rule would search it forever, while "
                        "better-than-Gaussian is the progress available there. "
                        "The cost is recall -- a truly-85%% ticket scores "
                        "below a 0.90 floor about a quarter of the time -- "
                        "which is acceptable only because you need ONE ticket, "
                        "not all of them.")
    p.add_argument("--keep_frac", type=float, default=1.0,
                   help="Fraction of survivors carried to the next tier. 0.5 "
                        "is plain sequential halving. 0.25 cuts the later "
                        "tiers by about 40%% and on goal T0's real tier-1 "
                        "distribution the top quarter still contains every "
                        "4/5 and 5/5 -- but this is a BUDGET knob, not a "
                        "correctness one: a smaller fraction can drop a good "
                        "ticket that had a bad five episodes. Default 1.0, "
                        "because under --prune_mode point the baseline floor "
                        "IS the selection rule and halving on top of it "
                        "discards candidates that already qualified.")
    p.add_argument("--prune_confidence", type=float, default=0.0,
                   help="Also drop candidates whose one-sided upper confidence "
                        "bound at this level is BELOW the Gaussian baseline -- "
                        "tickets that statistically cannot be as good as not "
                        "using one. Unlike a flat threshold this is calibrated "
                        "to the task's own baseline and to how many episodes "
                        "each candidate has actually run, so it carries no "
                        "risk of dropping a ticket that might be good. 0.90 is "
                        "a sensible value; on goal T0's tier 1 it takes the "
                        "survivors from 32 to 24. 0 disables it.")
    p.add_argument("--prune_floor", type=float, default=0.0,
                   help="Also drop candidates whose upper bound cannot reach "
                        "this ABSOLUTE rate -- for when the aim is a "
                        "high final number rather than merely beating "
                        "Gaussian. Still an upper bound, not the point "
                        "estimate: at five episodes a truly-85%% ticket scores "
                        "3/5 about 14%% of the time, and a rule that dropped "
                        "everything under 4/5 would discard it. The bound "
                        "keeps 3/5 (upper bound 0.888) and drops 2/5 (0.753). "
                        "Stacks with --prune_confidence, which uses the task's "
                        "own baseline instead of a fixed number; the stricter "
                        "of the two wins.")
    p.add_argument("--abort_below_ratio", type=float, default=0.0,
                   help="After tier 1, give up on a task whose "
                        "sd_between / sd_binomial_only is below this and move "
                        "to the next one. THE RATIO IS THE EFFECT SIZE; the "
                        "dispersion z is not -- z is driven by how many "
                        "episodes were run and does not separate the cases "
                        "(goal T0 z=9.85 succeeded, object T4 z=10.25 failed), "
                        "while the ratio does: 1.33 and 2.07 on the two tasks "
                        "that produced a working ticket against 1.07 on the "
                        "one that did not. THAT IS THREE DATA POINTS, so this "
                        "is off by default; 1.15 is the value those three "
                        "suggest. Worth enabling when a task costs 12 h, as "
                        "libero_10 does -- it turns a failed search into a "
                        "2 h answer instead of a 12 h one.")
    p.add_argument("--allow_zero_baseline", action="store_true",
                   help="Search on even when the Gaussian reference scores 0. "
                        "Without it the run aborts, because a zero baseline "
                        "almost always means the env is misconfigured rather "
                        "than the task being hard -- and finding that out "
                        "after 8 hours instead of 30 minutes is the expensive "
                        "version of the mistake.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-search tasks already in the bundle. Off by "
                        "default: a Colab session dies at 24 h and losing "
                        "finished tasks to a restart is the expensive mistake "
                        "this file is arranged around.")
    p.add_argument("--verbose", action="store_true",
                   help="Let eval_task print its per-call banner. Off by "
                        "default: the search makes hundreds of calls and the "
                        "banners bury the tier summaries, which are the only "
                        "lines worth watching.")
    a = p.parse_args()

    if a.init_state_offset + max(a.tiers * a.envs_per_tier
                                 + a.certify_layouts,
                                 a.baseline_layouts or 0) > 50:
        print(f"ERROR: tiers x envs_per_tier = "
              f"{a.tiers * a.envs_per_tier} layouts starting at "
              f"{a.init_state_offset} runs past the canonical 50 and would "
              f"plus {a.certify_layouts} certification layouts, "
              f"wrap into the reportable 0-19. Reduce --tiers, "
              f"--envs_per_tier, --certify_layouts, or "
              f"--init_state_offset.", file=sys.stderr)
        return 1

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from checkpoint_utils import resolve_checkpoint
    # One call, the same one eval_libero.main() makes. Copying the patches
    # individually is how this script shipped two separate bugs.
    ev.setup_libero_env(a.control_freq, a.render_gpu, a.stock_init)
    ckpt = resolve_checkpoint(a.checkpoint, for_resume=False)
    policy = ev.load_policy(ckpt, device, a.num_inference_steps,
                            n_action_steps=a.n_action_steps,
                            vision_input_size=a.vision_input_size)
    pre, post = ev.load_processors(ckpt, device, a.dataset_id)
    cams = list(policy.config.cameras_for_vision_state_concat) \
        if hasattr(policy.config, "cameras_for_vision_state_concat") else []
    if not hasattr(getattr(policy, "model", None), "sample_actions"):
        print(f"ERROR: {type(policy).__name__} has no .model.sample_actions, so "
              f"there is nowhere to inject a ticket. Ticket search is "
              f"implemented for the wilro_moe family.", file=sys.stderr)
        return 1
    H = int(policy.config.horizon)
    D = int(policy.config.action_dim)
    # A ticket is only valid for the inference config it was searched under,
    # and a silent mismatch looks exactly like "the method does not work".
    infcfg = ev.inference_config(policy, a.control_freq, a.max_episode_steps,
                                 a.stock_init)
    ev.report_inference_config(
        infcfg, "these must match the eval command you will report with; "
                "eval refuses a ticket whose " + "/".join(ev.MUST_MATCH) +
                " differ")

    if a.require_perfect:
        span = a.tiers * a.envs_per_tier
        covers_eval = a.init_state_offset == 0 and span >= 20
        print(f"[perfect] floor 1.0 at every tier; searching layouts "
              f"{a.init_state_offset}-{a.init_state_offset + span - 1}", flush=True)
        if covers_eval:
            print("  THE REPORTED LAYOUTS 0-19 ARE INSIDE THIS SEARCH. A ticket\n"
                  "  that survives is VERIFIED to solve them -- that is a fact,\n"
                  "  not an estimate, and it is the only route to a "
                  "deterministic\n"
                  "  100%. It is also selection over the canonical set, so the\n"
                  "  claim to publish is 'solves all N canonical layouts', "
                  "never\n"
                  "  'generalises': no unseen init state was tested.",
                  flush=True)
        elif span < 50:
            print(f"  layouts 0-19 are NOT in this search, so a survivor is\n"
                  f"  verified on {span} layouts and still only ESTIMATED on "
                  f"the\n  reported ones. 30/30 leaves P(20/20) at 61%.",
                  flush=True)
    _nb = a.baseline_layouts or (5 if a.require_perfect
                                 else a.tiers * a.envs_per_tier)
    if not a.require_perfect and _nb < a.tiers * a.envs_per_tier:
        print(f"\n*** --baseline_layouts {_nb} is below tiers x envs_per_tier "
              f"({a.tiers * a.envs_per_tier}), which costs two things.\n"
              f"    --stratify_layouts is SKIPPED, so each tier gets whichever "
              f"layouts sit at its ids;\n"
              f"    per-task difficulty varies enormously -- libero_10 T0 reads "
              f"90% on 20-24 and\n"
              f"    40% on 25-29 -- so tier 1, the tier that filters every "
              f"candidate, draws its\n"
              f"    floor by position.\n"
              f"    And the pooled baseline describes only the layouts it "
              f"measured, so the final\n"
              f"    'winner X% vs Gaussian Y%' verdict is computed against a "
              f"Y that omits the rest.\n"
              f"    Read --certify_layouts instead: it runs the ticket and "
              f"Gaussian head to head on\n"
              f"    layouts that took no part in the selection, so it does not "
              f"depend on this.\n", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # Batches per tier are M x ceil(alive / num_envs), NOT ceil(M x alive /
    # num_envs): score() loops layout-outer, candidate-inner, so a tier with 4
    # survivors still pays a full batch per layout. Summing episodes and
    # dividing once undercounted every deep tier.
    n_base = a.baseline_layouts or (a.tiers * a.envs_per_tier)
    batches, _a = n_base, a.tickets
    for _t in range(a.tiers):
        batches += a.envs_per_tier * -(-_a // a.num_envs)
        if _t < a.tiers - 1:
            _a = max(1, int(_a * a.keep_frac))
    hours = batches * 12 * ((a.max_episode_steps or 300) / 300) / 60
    # SAY WHICH MODE THIS IS. A --cycle 4 run and a single-ticket run produce
    # the same log otherwise, and the first-layout pass rate is the only tell:
    # about 6-12% at m=1 on libero_10 against about 65% at m=4. Reading a 6%
    # and concluding "cycles do not help" when --cycle was simply not passed
    # is a cheap mistake to make and an expensive one to find.
    print(f"NOISE: " + ("one constant vector per chunk (m=1, the plain golden "
                        "ticket)" if a.cycle == 1 else
                        f"a cycle of {a.cycle} vectors, chunk k uses "
                        f"t[k mod {a.cycle}]"), flush=True)
    print(f"candidate shape "
          + (f"({H}, {D})" if a.cycle == 1 else f"({a.cycle}, {H}, {D})")
          + f" = {a.cycle * H * D} dims\n"
          f"{a.tickets} candidates, {a.tiers} tiers x {a.envs_per_tier} layouts "
          f"from id {a.init_state_offset}, baseline on {n_base} layouts\n"
          f"~{batches} batches of {a.num_envs} ({n_base} of them baseline)\n"
          f"WORST CASE: --prune_mode {a.prune_mode} and --exact_prune drop "
          f"candidates the moment they cannot reach the floor, and on a "
          f"high-baseline task that removes most of tier 1.\n"
          f"BATCHES ARE THE COST, NOT EPISODES: a batch runs until its slowest "
          f"env finishes and most reach the cap. At this project's measured "
          f"12 min/batch at cap 300, that is ~{hours:.1f} h for this task.",
          flush=True)

    results = {}
    # --overwrite USED TO DISABLE RESUME. Every `not a.overwrite` guard below
    # covered both the bundle skip and the progress file, so an --overwrite run
    # killed at hour 10 of 12 restarted from zero, forever, on a platform whose
    # sessions end at 24 h. Progress written BEFORE this process started is the
    # run being overwritten and is discarded; progress written after is this
    # run's own checkpoint and is resumed.
    t_start = time.time()
    from lerobot.envs.libero import LiberoEnv, _get_suite
    for suite_name in a.suites:
        suite = _get_suite(suite_name)
        n_tasks = getattr(suite, "n_tasks", None) or len(suite.tasks)
        ids = a.task_ids if a.task_ids is not None else list(range(n_tasks))
        for tid in ids:
            t0 = time.time()
            if not a.overwrite:
                try:
                    done, dmeta = tb.load_bundle(out)
                    _k = tb.key(suite_name, tid)
                    _m = dmeta.get(_k, {}) if _k in done else None
                    # A BANKED TICKET IS NOT THE SAME AS A WON ONE. Search
                    # banks whatever it finished, including tasks it could not
                    # beat Gaussian on, and the old message read the same for
                    # both: object T5 at 53% against an 85% baseline printed
                    # "already in the bundle" next to T9's 15/15. Those need
                    # different follow-ups -- one is done, the other needs an
                    # order of magnitude more candidates.
                    if _m is not None and _m.get("beats_baseline") is False \
                            and a.retry_failed:
                        print(f"\n=== {suite_name} task {tid}: in the bundle "
                              f"but marked NOT BETTER than Gaussian "
                              f"({_m.get('search_success', '?')} vs "
                              f"{_m.get('baseline_search', '?')}) -- "
                              f"--retry_failed, re-searching ===", flush=True)
                    elif _m is not None and _m.get("beats_baseline") is False:
                        print(f"\n=== {suite_name} task {tid}: in the bundle "
                              f"but NOT BETTER than Gaussian "
                              f"({_m.get('search_success', '?')} vs "
                              f"{_m.get('baseline_search', '?')}). Skipping. "
                              f"Eval ignores it; redo with --retry_failed and "
                              f"more --tickets ===", flush=True)
                        continue
                    elif _m is not None:
                        print(f"\n=== {suite_name} task {tid}: already in the "
                              f"bundle, skipping (--overwrite to redo) ===",
                              flush=True)
                        continue
                except FileNotFoundError:
                    pass
            print(f"\n=== {suite_name} task {tid} ===", flush=True)
            # Built ONCE per task and handed to every eval_task call.
            # Construction takes seconds per env and the search makes hundreds
            # of calls, so building them per call would dominate the run.
            n_par = a.num_envs
            print(f"  building {n_par} envs...", end="", flush=True)
            _tb = time.time()
            envs = [LiberoEnv(task_suite=suite, task_id=tid,
                              task_suite_name=suite_name,
                              obs_type="pixels_agent_pos",
                              init_states=True, episode_index=0)
                    for _ in range(n_par)]
            print(f" {time.time() - _tb:.0f}s", flush=True)
            # Candidates are fixed up front so every tier scores the SAME
            # tickets, and the baseline (all-Gaussian) is not among them: it is
            # measured separately, at the same layouts, as ticket id -1.
            # Within-task progress, rewritten after EVERY tier. A task is
            # hours; losing it at tier 3 to a 24 h cutoff is what this guards
            # against. `cands` is saved too, so a resumed run scores the SAME
            # candidates -- otherwise the accumulated wins/runs would describe
            # tickets that no longer exist.
            prog = out / f"_progress_{suite_name}_t{tid}.npz"
            if prog.exists() and a.overwrite and prog.stat().st_mtime < t_start:
                print(f"  --overwrite: discarding {prog.name} from the "
                      f"previous run", flush=True)
                prog.unlink()
            resume_ok = prog.exists() and (not a.overwrite
                                           or prog.stat().st_mtime >= t_start)
            if resume_ok:
                # Progress from a differently-configured run is WORSE than no
                # progress: it carries that run's counts and its base_done
                # flag, so the Gaussian reference is never re-measured and the
                # zero-baseline guard fires on stale numbers without a single
                # new rollout. That is exactly what happened on the first run
                # after the patch_lerobot_libero and n_action_steps fixes --
                # the env was finally right and the abort still read 0/50 from
                # the broken run's file.
                # STRICTER than ev.MUST_MATCH, and deliberately so. MUST_MATCH
                # asks "is this ticket still valid for this policy", and a cap
                # is not part of that: a ticket searched at 150 steps is fine to
                # eval at 300, it was merely selected for speed. Resuming counts
                # asks a different question -- "are these wins commensurable" --
                # and there the cap matters, because adding 2 episodes at cap
                # 150 to 2 already banked at cap 300 puts two different success
                # rates in one numerator, and leaves base_done set so the
                # baseline keeps the generous cap while the candidates get the
                # strict one. That biases the prune floor against every
                # survivor.
                # The search's OWN geometry has to match too, and none of it
                # is in infcfg. `layout = init_state_offset + tier *
                # envs_per_tier + k`, so resuming goal T2's M=4 tier 1 under
                # M=2 put tier 2 back on layouts 22-23, which tier 1 had
                # already scored: the survivor's 6/6 was four distinct layouts
                # and two repeats, and sequential halving's whole argument is
                # that the tiers are disjoint.
                _geom = {"envs_per_tier": a.envs_per_tier, "tickets": a.tickets,
                         "cycle": a.cycle,
                         "init_state_offset": a.init_state_offset,
                         "baseline_layouts": (a.baseline_layouts or
                                              a.tiers * a.envs_per_tier),
                         "prune_mode": a.prune_mode}
                _resume_match = ev.MUST_MATCH + ("max_episode_steps",)
                _z = np.load(prog, allow_pickle=True)
                if "infcfg" not in _z.files:
                    _bad = {"(unstamped)": ("pre-dates the config stamp", "")}
                else:
                    _was = json.loads(str(_z["infcfg"]))
                    _bad = {k: (_was.get(k), infcfg[k]) for k in _resume_match
                            if _was.get(k) != infcfg[k]}
                    _wasg = (json.loads(str(_z["geom"]))
                             if "geom" in _z.files else {})
                    _bad.update({k: (_wasg.get(k, "(unstamped)"), v)
                                 for k, v in _geom.items()
                                 if _wasg.get(k, v) != v})
                if _bad:
                    print(f"  DISCARDING {prog.name}: written under "
                          + ", ".join(f"{k}={w}{f' (now {n_})' if n_ != '' else ''}"
                                      for k, (w, n_) in _bad.items())
                          + " -- starting this task fresh.", flush=True)
                    prog.unlink()
            if prog.exists():
                z = np.load(prog)
                cands, wins, runs = z["cands"], z["wins"], z["runs"]
                alive, first_tier = [int(x) for x in z["alive"]], int(z["next_tier"])
                base_w0, base_r0 = float(z["base_w"]), float(z["base_r"])
                base_lw0 = (z["base_lw"].astype(float) if "base_lw" in z.files
                            else None)
                base_lr0 = (z["base_lr"].astype(float) if "base_lr" in z.files
                            else None)
                # Kept in the progress file because a run killed BETWEEN the
                # last tier's save and the bundle write resumes with an empty
                # tier loop: the ticket is recovered correctly but nothing
                # would re-read the task string.
                desc0 = str(z["desc"]) if "desc" in z.files else None
                # Layouts already finished INSIDE next_tier, and whether the
                # Gaussian reference has run. Tier 1 is 3.7 of a task's 6.1
                # hours, so checkpointing only between tiers leaves 3.7 hours
                # exposed to a Colab cutoff; per layout it is about 40 min.
                start_k0 = int(z["done_k"]) if "done_k" in z.files else 0
                base_done0 = bool(z["base_done"]) if "base_done" in z.files else False
                # How many baseline layouts are finished. Older files only say
                # "done / not done", so an unfinished one there restarts the
                # baseline -- which is what it used to do in every case.
                base_k0 = int(z["base_k"]) if "base_k" in z.files else None
                lay_order0 = (z["lay_order"].astype(int)
                              if "lay_order" in z.files else None)
                if start_k0 > 0 and float(runs.sum()) == 0.0:
                    # Written by the pre-fix score_baseline: the tier is marked
                    # part-done while no candidate has run an episode. Heal it
                    # rather than skipping the tier and halving on all-zeros.
                    print(f"  progress claims {start_k0} layouts done but no "
                          f"candidate has run an episode -- resetting to the "
                          f"start of the tier (pre-fix file)", flush=True)
                    start_k0 = 0
                print(f"  resuming from {prog.name}: tier {first_tier + 1}, "
                      f"{len(alive)} candidates still alive", flush=True)
            else:
                # SEEDED PER TASK, not from one stream shared by the run.
                # Tasks already in the bundle `continue` before this line, so
                # a shared stream handed task 4 the candidates task 0 drew on
                # the previous pass -- which candidate set a task got depended
                # on what else happened to be finished, and "delete the
                # progress and run it again" was neither a no-op nor a fresh
                # draw. Keyed on the task, --seed is now the only thing that
                # changes the candidates.
                rng = np.random.default_rng(
                    [a.seed, zlib.crc32(suite_name.encode()), tid])
                shape = ((a.tickets, H, D) if a.cycle == 1
                         else (a.tickets, a.cycle, H, D))
                cands = rng.standard_normal(shape).astype(np.float32)
                alive, first_tier = list(range(a.tickets)), 0
                wins = np.zeros(a.tickets); runs = np.zeros(a.tickets)
                base_w0 = base_r0 = 0.0
                base_lw0 = base_lr0 = None
                base_k0 = None
                lay_order0 = None
                desc0 = None
                start_k0, base_done0 = 0, False

            # PER LAYOUT, not pooled. The floor is a hard gate now, and the
            # tiers run on DIFFERENT layouts: if the Gaussian goes 10/10 on
            # layout 24 and 6/10 on 25, tier 3's real reference is 0.80, and
            # gating it on the pooled 0.96 kills every candidate on the harder
            # half of the search. Indexed by layout - init_state_offset.
            # Under --require_perfect the floor does not come from the
            # baseline, so the baseline is only the env sanity check that
            # caught patch_lerobot_libero and n_action_steps. Five layouts is
            # enough for that; fifty would be 50 batches of nothing.
            n_base_lay = (a.baseline_layouts
                          or (5 if a.require_perfect
                              else a.tiers * a.envs_per_tier))
            base_lw = (base_lw0 if base_lw0 is not None and
                       len(base_lw0) == n_base_lay else np.zeros(n_base_lay))
            base_lr = (base_lr0 if base_lr0 is not None and
                       len(base_lr0) == n_base_lay else np.zeros(n_base_lay))
            # PER LAYOUT rather than a done/not-done flag. 15 baseline layouts
            # is 90 minutes, and checkpointing only at the end left all of it
            # exposed: a cutoff mid-baseline resumed by running the whole
            # baseline again, which on a short Colab lease never finishes.
            base_k = [base_k0 if base_k0 is not None
                      else (n_base_lay if base_done0 else 0)]
            # WHICH LAYOUT GOES IN WHICH TIER. Identity until the baseline has
            # run; --stratify_layouts then deals them out so no tier draws all
            # the easy ones. libero_10 T0 measured 45/50 on layouts 20-24 and
            # 20/50 on 25-29 -- a 50 point spread inside one task -- and tier
            # 1, the widest tier, happened to get the easy five and a 5/5
            # floor. Which five land in tier 1 should not decide the search.
            n_srch = a.tiers * a.envs_per_tier
            lay_order = (lay_order0 if lay_order0 is not None
                         and len(lay_order0) == n_srch
                         else np.arange(n_srch))

            def _save(tier, done_k, alive_now):
                np.savez(prog, cands=cands, wins=wins, runs=runs,
                         alive=np.array(alive_now, dtype=np.int64),
                         next_tier=tier, done_k=done_k,
                         base_w=base_w[0], base_r=base_r[0],
                         base_done=base_k[0] >= n_base_lay,
                         base_k=base_k[0], desc=np.array(desc or ""),
                         lay_order=np.asarray(lay_order, dtype=np.int64),
                         base_lw=base_lw, base_lr=base_lr,
                         infcfg=np.array(json.dumps(infcfg)),
                         geom=np.array(json.dumps(
                             {"envs_per_tier": a.envs_per_tier,
                              "tickets": a.tickets,
                              "cycle": a.cycle,
                              "init_state_offset": a.init_state_offset,
                              "baseline_layouts": n_base_lay,
                              "prune_mode": a.prune_mode})))

            def score(idx_list, tier, start_k=0):
                """One batch = n_par CANDIDATES on ONE layout.

                Wall clock is set by the number of BATCHES, not episodes: a
                batch runs until its slowest env finishes, and at a 70%
                success rate 97% of ten-env batches reach the cap. Scoring one
                candidate per batch therefore burns a whole batch on five
                episodes. A per-env ticket puts a different candidate in every
                env against the same layout -- which is also the fairest
                comparison available: identical problem, identical seed, only
                the ticket differs.
                """
                nonlocal desc
                M, floor = a.envs_per_tier, floor_for(tier)
                n_end = (tier + 1) * M            # runs each survivor ends on
                idx_list = list(idx_list)
                for k in range(start_k, a.envs_per_tier):
                    layout = a.init_state_offset + int(
                        lay_order[tier * a.envs_per_tier + k])
                    for g0 in range(0, len(idx_list), n_par):
                        grp = idx_list[g0:g0 + n_par]
                        tk = torch.from_numpy(
                            np.stack([cands[i] for i in grp])).to(device)
                        if a.cycle == 1:
                            policy.model._noise_ticket = tk
                        else:
                            # (n_par, m, H, D): a different cycle per env, so
                            # one batch still scores n_par candidates against
                            # one layout.
                            policy.model._noise_cycle = tk
                            policy.model._noise_ticket = None
                        sink = (contextlib.nullcontext() if a.verbose
                                else contextlib.redirect_stdout(io.StringIO()))
                        with sink:
                            _, _, _, _, desc, ep_ok = ev.eval_task(
                                policy, pre, post, suite, suite_name, tid,
                                len(grp), len(grp), device,
                                a.max_episode_steps, a.seed, cams,
                                envs=envs[:len(grp)],
                                init_state_offset=layout, init_state_stride=0)
                        for j, i in enumerate(grp):
                            wins[i] += ep_ok[j]; runs[i] += 1
                    # Arithmetic elimination, not a statistical call: with
                    # M - (k+1) layouts left a candidate can finish no higher
                    # than (wins + remaining) / n_end, and if that is already
                    # under the floor the rest of the tier is wasted on it.
                    # NOT in the final tier. That tier ranks the survivors
                    # rather than filtering them -- the bank-or-not call is
                    # made afterwards on the CUMULATIVE rate against the pooled
                    # baseline -- so eliminating against the final tier's local
                    # floor can throw away the candidate that would have won.
                    if a.exact_prune and a.prune_mode == "point" and floor > 0 \
                            and k + 1 < M and tier < a.tiers - 1:
                        rem = M - (k + 1)
                        live = [i for i in idx_list
                                if (wins[i] + rem) / n_end >= floor - 1e-9]
                        if not live:              # never prune to nothing
                            live = [max(idx_list, key=lambda i: wins[i])]
                            print(f"    layout {k + 1}/{M}: NO candidate can "
                                  f"still reach the {floor:.0%} floor -- "
                                  f"carrying the best one ({int(wins[live[0]])}"
                                  f"/{int(runs[live[0]])}) so the tier still "
                                  f"returns something", flush=True)
                        elif len(live) < len(idx_list):
                            print(f"    layout {k + 1}/{M}: "
                                  f"{len(idx_list)} -> {len(live)} "
                                  f"(cannot reach {floor:.0%} any more)",
                                  flush=True)
                        # THE CHEAP VERDICT, 40 minutes in instead of 4
                        # hours. One layout scored by EVERY candidate gives
                        # the per-layout pass rate, and the tier's floor says
                        # how many of M a candidate has to win; together they
                        # say how many candidates the task needs. object T5
                        # read 27% against a 5-of-5 floor -- one in 700 -- and
                        # 64 candidates spent 3.7 hours confirming it.
                        if tier == 0 and k == 0 and idx_list:
                            need = int(np.ceil(floor * M - 1e-9))
                            # From the wins, not from the survivor count: a low
                            # floor eliminates nobody after one layout and the
                            # survivor count would read 100%.
                            phat = float(np.mean([wins[i] for i in idx_list]))
                            q = sum(comb(M, j) * phat ** j * (1 - phat) ** (M - j)
                                    for j in range(need, M + 1))
                            msg = (f"    layout 1 pass rate {phat:.0%} vs "
                                   f"Gaussian {floor:.0%}; the floor needs "
                                   f"{need}/{M}, ")
                            if q <= 0:
                                print(msg + "which nothing at this rate "
                                      "reaches -- more candidates will not "
                                      "help, the ticket effect is the "
                                      "problem", flush=True)
                            else:
                                print(msg + f"so about {1 / q:.0f} candidates "
                                      f"per survivor", flush=True)
                                if a.tickets * q < 0.5:
                                    print(f"    -> --tickets {a.tickets} "
                                          f"expects {a.tickets * q:.2f} "
                                          f"survivors. Killing this now and "
                                          f"restarting with ~{int(1.5 / q)} "
                                          f"costs less than finishing.",
                                          flush=True)
                        idx_list = live
                    # `alive` is not touched until score() returns, so saving
                    # it here records the tier's live list -- which is what a
                    # mid-tier resume has to continue from.
                    _save(tier, k + 1, idx_list)
                return desc, idx_list

            def score_baseline(tier, cand_k):
                """The Gaussian reference, from the first search layout on.

                `cand_k` is how many layouts the CANDIDATES have finished, not
                how many the baseline just ran. Saving the baseline's count
                here claimed the candidates were done with the tier while they
                had not started it, so a run that died between the baseline and
                the end of scoring resumed with start_k == envs_per_tier and
                skipped tier 1's candidate scoring entirely -- every ticket
                then carried 0/0 into the halving.
                """
                if base_k[0] >= n_base_lay:
                    return
                policy.model._noise_ticket = None
                policy.model._noise_cycle = None
                # NOT offset by `tier`. The baseline has to cover every layout
                # the search will use, and on a resume at tier > 0 the old
                # `tier * envs_per_tier + k` slid the whole reference off the
                # layouts tier 1 had been measured against.
                n_lay = n_base_lay
                for k in range(base_k[0], n_lay):
                    layout = a.init_state_offset + k
                    sink = (contextlib.nullcontext() if a.verbose
                            else contextlib.redirect_stdout(io.StringIO()))
                    with sink:
                        n_ok, n_ep, _, _, _, _ = ev.eval_task(
                            policy, pre, post, suite, suite_name, tid,
                            n_par, n_par, device, a.max_episode_steps,
                            a.seed, cams, envs=envs,
                            init_state_offset=layout, init_state_stride=0)
                    base_w[0] += n_ok; base_r[0] += n_ep
                    base_lw[k] += n_ok; base_lr[k] += n_ep
                    base_k[0] = k + 1
                    _save(tier, cand_k, alive)
                if a.stratify_layouts and n_base_lay >= n_srch:
                    # Hardest first, dealt round robin, so every tier gets a
                    # comparable mix and the floors come out within a few
                    # points of each other instead of 90% against 40%.
                    rate = np.where(base_lr[:n_srch] > 0,
                                    base_lw[:n_srch] / np.maximum(base_lr[:n_srch], 1),
                                    0.0)
                    order = list(np.argsort(rate, kind="stable"))
                    dealt = [[] for _ in range(a.tiers)]
                    for j, L in enumerate(order):
                        dealt[j % a.tiers].append(int(L))
                    lay_order[:] = [L for grp in dealt for L in grp]
                    for t in range(a.tiers):
                        seg = lay_order[t * a.envs_per_tier:
                                        (t + 1) * a.envs_per_tier]
                        f = (base_lw[seg].sum() / max(base_lr[seg].sum(), 1))
                        print(f"    tier {t + 1} layouts "
                              + " ".join(str(a.init_state_offset + int(L))
                                         for L in sorted(seg))
                              + f"  Gaussian {f:.0%}", flush=True)
                _save(tier, cand_k, alive)

            base_w, base_r = [base_w0], [base_r0]

            def tier_floor(tier):
                """Gaussian over every layout a survivor of `tier` has RUN.

                NOT the layouts of this tier alone. wins/runs is cumulative
                across tiers, so comparing it to one tier's local rate puts a
                numerator spanning ten layouts over a denominator describing
                five. On libero_10 T0 that meant a 3/10 record -- layouts
                20-29, where Gaussian scores 65 -- judged against tier 2's
                local 40%.
                """
                n = min((tier + 1) * a.envs_per_tier, len(lay_order))
                seg = lay_order[:n]
                seg = seg[seg < len(base_lr)]
                seg_r = base_lr[seg].sum() if len(seg) else 0.0
                if seg_r > 0:
                    return float(base_lw[seg].sum() / seg_r)
                return float(base_w[0] / base_r[0]) if base_r[0] > 0 else 0.0

            def floor_for(tier):
                if a.require_perfect:
                    return 1.0
                f = 0.0
                if (a.prune_confidence > 0 or a.prune_mode == "point") \
                        and base_r[0] > 0:
                    f = tier_floor(tier)
                return max(f, a.prune_floor)

            desc, abandoned = desc0, False
            floor_empty = False
            for tier in range(first_tier, a.tiers):
                _seg = lay_order[tier * a.envs_per_tier:
                                 (tier + 1) * a.envs_per_tier]
                print(f"  tier {tier + 1}/{a.tiers}: {len(alive)} candidates "
                      f"on {a.envs_per_tier} layouts (ids "
                      + " ".join(str(a.init_state_offset + int(L))
                                 for L in sorted(_seg)) + ")", flush=True)
                if tier == 0:
                    # BEFORE the candidates, not after. This is the only cheap
                    # check that the env is set up the way the reported evals
                    # set it up, and it has to happen before hours are spent.
                    score_baseline(tier, start_k0 if tier == first_tier else 0)
                    if a.reported_rate is not None and base_r[0] > 0:
                        _br = base_w[0] / base_r[0]
                        _gap = a.reported_rate - _br
                        print(f"    calibration: the per-chunk draw scores "
                              f"{_br:.0%} on these search layouts against "
                              f"{a.reported_rate:.0%} on the reported ones, a "
                              f"gap of {_gap:+.0%}.", flush=True)
                        if abs(_gap) > 0.15:
                            # A FACT, NOT A PREDICTION. This was a warning
                            # that said to consider stopping, built on two
                            # tasks: goal T9 at +31% lost 15 points and goal
                            # T6 at -21% gained 15. Two more broke it --
                            # libero_10 T0 at ~0% gained 25 and T1 at +4%
                            # lost 20 -- so the gap does not separate the
                            # cases and nothing here forecasts the outcome.
                            # What does catch a bad ticket is measuring the
                            # per-chunk draw on the reported layouts, two
                            # batches, after the search. That has caught all
                            # three tickets that were costing points.
                            print(f"    the two layout sets differ in "
                                  f"difficulty by {abs(_gap):.0%}; the floor "
                                  f"below is calibrated to {_br:.0%}, not to "
                                  f"the {a.reported_rate:.0%} the eval asks "
                                  f"for.\n"
                                  f"    Across four tasks this gap has not "
                                  f"predicted whether the resulting ticket "
                                  f"helps or hurts, so it is\n"
                                  f"    not a reason to stop. Run the "
                                  f"per-chunk control on layouts 0-19 after "
                                  f"the search and compare there.\n",
                                  flush=True)
                    if base_w[0] == 0 and not a.allow_zero_baseline:
                        for _e in envs:
                            try:
                                _e.close()
                            except Exception:
                                pass
                        raise SystemExit(
                            f"\nGaussian baseline scored 0/{base_r[0]:.0f} on "
                            f"layouts {a.init_state_offset}-"
                            f"{a.init_state_offset + n_base_lay - 1} of "
                            f"{suite_name} task {tid}.\n"
                            f"This policy is not at 0% on this task, so the "
                            f"env is almost certainly not the one the evals "
                            f"use. Check that patch_lerobot_libero and "
                            f"--control_freq {a.control_freq} match the eval "
                            f"command, and that --max_episode_steps "
                            f"{a.max_episode_steps} is not cutting successes "
                            f"off.\nSearching for a ticket on a task the "
                            f"policy cannot do at all learns nothing. "
                            f"--allow_zero_baseline to override.")
                desc, alive = score(alive, tier,
                                    start_k0 if tier == first_tier else 0)
                rate = np.where(runs > 0, wins / np.maximum(runs, 1), -1.0)
                alive = sorted(alive, key=lambda i: -rate[i])
                if tier < a.tiers - 1:
                    n_before = len(alive)
                    # FLOOR FIRST, cap second. The old order halved by rank and
                    # only then applied the floor, so a candidate that met the
                    # baseline could be cut by keep_frac before the rule that
                    # decides membership ever looked at it.
                    floor = floor_for(tier)
                    if floor > 0:
                        conf = a.prune_confidence or 0.90
                        if a.prune_mode == "point":
                            kept = [i for i in alive
                                    if runs[i] > 0
                                    and wins[i] / runs[i] >= floor - 1e-9]
                            how = (f"below the {floor:.0%} Gaussian rate "
                                   f"over the {(tier + 1) * a.envs_per_tier} "
                                   f"layouts they have run")
                        else:
                            kept = [i for i in alive
                                    if binom_ub(int(wins[i]), int(runs[i]),
                                                conf) >= floor]
                            how = (f"whose {conf:.0%} upper bound is below the "
                                   f"{floor:.0%} baseline")
                        dropped = len(alive) - len(kept)
                        if kept:
                            alive = kept
                            if dropped:
                                print(f"    pruned {dropped} {how}", flush=True)
                        else:
                            # Loud, because it used to be silent: the tier
                            # carries its best candidate purely so the task
                            # still produces a ticket, and that ticket has NOT
                            # beaten Gaussian.
                            alive = alive[:1]
                            floor_empty = True
                            print(f"    NO candidate met the {floor:.0%} "
                                  f"floor. Carrying the best "
                                  f"({int(wins[alive[0]])}/"
                                  f"{int(runs[alive[0]])}) so the task still "
                                  f"finishes, but it is NOT better than "
                                  f"Gaussian and should not be banked.",
                                  flush=True)
                    if a.keep_frac < 1.0:
                        alive = alive[:max(1, int(len(alive) * a.keep_frac))]
                    print(f"    {n_before} -> {len(alive)} carried forward",
                          flush=True)
                    if floor_empty and a.abandon_on_empty_floor:
                        # The remaining tiers cannot rescue it. Whatever is
                        # carried forward is already below the floor, the
                        # bank-or-not check at the end will read
                        # NOT BETTER than Gaussian, and object T5 spent an
                        # hour on tiers 2 and 3 arriving there.
                        print(f"    ABANDONING this task: no candidate met "
                              f"tier {tier + 1}'s floor, so the remaining "
                              f"{a.tiers - tier - 1} tiers can only confirm "
                              f"it. No ticket is banked.", flush=True)
                        abandoned = True
                        break
                top = alive[0]
                if runs[top] == 0:
                    raise SystemExit(
                        f"the best candidate has run 0 episodes, so the tier "
                        f"scored nothing and the halving just kept the first "
                        f"{len(alive)} indices. Delete "
                        f"{prog.name} and restart this task.")
                print(f"    best so far: ticket {top} "
                      f"{wins[top]:.0f}/{runs[top]:.0f} = "
                      f"{100 * rate[top]:.0f}%   "
                      f"(baseline Gaussian {base_w[0]:.0f}/{base_r[0]:.0f})",
                      flush=True)
                _save(tier + 1, 0, alive)
                # The full distribution, not just the winner. After tier 1 the
                # SPREAD across candidates is what says whether this policy is
                # steerable at all, and the winner of a 5-episode tier is
                # mostly luck: 128 identical tickets at p=0.7 throw ~21 perfect
                # scores by chance.
                # THE TEST NEEDS EQUAL RUNS PER CANDIDATE, and --exact_prune
                # takes that away: a candidate is stopped BECAUSE it lost, so
                # the survivors' rates are truncated upward and the spread
                # collapses. object T5 printed sd_between 0.000, dispersion
                # 0.62 and "not steerable by the initial noise" off a sample
                # where 47 candidates had one episode, 10 had two and one had
                # fifteen. That conclusion had no support: the only layout
                # every candidate ran gives w/r in {0, 1}, which cannot carry
                # between-candidate variance at all.
                r_alive = [runs[i] for i in range(a.tickets) if runs[i] > 0]
                truncated = (a.exact_prune and a.prune_mode == "point"
                             and len(set(r_alive)) > 1)
                disp = dispersion(wins, runs)
                if truncated:
                    print(f"    dispersion test SKIPPED: --exact_prune stopped "
                          f"candidates on their results, so runs per candidate "
                          f"range {min(r_alive)}-{max(r_alive)} and the spread "
                          f"is truncated, not measured. The survival curve "
                          f"above is the readable signal.", flush=True)
                    disp = None
                if disp and "dispersion" in disp:
                    verdict = ("TICKETS DIFFER -- worth continuing"
                               if disp["z"] >= 3 else
                               "suggestive, needs more layouts per ticket"
                               if disp["z"] >= 1.5 else
                               "NO real spread: the observed range is what "
                               "binomial noise alone produces. This policy is "
                               "not steerable by the initial noise")
                    # NORMALISED TO 5 LAYOUTS PER TIER. sd_binomial is
                    # sqrt(pq/M), so the raw ratio is proportional to sqrt(M)
                    # and a threshold calibrated at M=5 is 1.58x too strict at
                    # M=2. goal T0 -- the only task whose ticket has held up
                    # out of sample -- reads 1.33 at M=5 and would read 0.84
                    # at M=2, i.e. a gate at 1.15 would have killed it.
                    raw = (disp["sd_between"] / disp["sd_binomial_only"]
                           if disp["sd_binomial_only"] > 0 else float("inf"))
                    ratio = raw * (5.0 / max(a.envs_per_tier, 1)) ** 0.5
                    print(f"    dispersion {disp['dispersion']:.2f} over "
                          f"{disp['n']} candidates (1.00 = interchangeable)  "
                          f"z={disp['z']:+.1f}  "
                          f"sd_between {disp['sd_between']:.3f} vs "
                          f"binomial {disp['sd_binomial_only']:.3f}  "
                          f"ratio {ratio:.2f}"
                          + (f" (raw {raw:.2f} at M={a.envs_per_tier}, "
                             f"normalised to M=5)" if a.envs_per_tier != 5
                             else "") + "\n"
                          f"    -> {verdict}", flush=True)
                    _br = base_w[0] / max(base_r[0], 1)
                    if (tier == 0 and a.abort_below_ratio > 0 and _br >= 0.95):
                        print(f"    gate SKIPPED: the Gaussian baseline is "
                              f"{_br:.0%}, so there is no room for "
                              f"between-ticket variance to show and the ratio "
                              f"cannot discriminate. Searching on -- but a task "
                              f"this close to ceiling has no headroom to win "
                              f"either, and should probably not be in "
                              f"--task_ids.", flush=True)
                    elif (tier == 0 and a.abort_below_ratio > 0
                            and ratio < a.abort_below_ratio):
                        print(f"    ABANDONING this task: effect size "
                              f"{ratio:.2f} < --abort_below_ratio "
                              f"{a.abort_below_ratio:g}. Almost all of the "
                              f"spread between candidates is binomial noise, "
                              f"which is what a failed search looks like at "
                              f"tier 1 (object T4 read 1.07 and cost 12 h to "
                              f"confirm). No ticket is banked.", flush=True)
                        abandoned = True
                        break
                (out / f"scores_{suite_name}_t{tid}.json").write_text(json.dumps(
                    {"tier": tier + 1, "task": desc,
                     "baseline": f"{base_w[0]:.0f}/{base_r[0]:.0f}",
                     "dispersion_test": disp,
                     "candidates": {str(i): [int(wins[i]), int(runs[i])]
                                    for i in range(a.tickets) if runs[i] > 0}},
                    indent=1))

            if abandoned:
                # No ticket, and the progress file is kept: the candidates are
                # still on disk if a later run wants to resume with a lower
                # threshold or more tiers.
                results[f"{suite_name}_t{tid}"] = {
                    "task": desc, "abandoned_at_tier": 1,
                    "reason": ("no candidate met the tier floor"
                               if floor_empty else
                               f"effect size below --abort_below_ratio "
                               f"{a.abort_below_ratio:g}"),
                    "baseline_search": f"{base_w[0]:.0f}/{base_r[0]:.0f}",
                    "minutes": round((time.time() - t0) / 60, 1)}
                (out / "search_summary.json").write_text(json.dumps(results, indent=1))
                for e_ in (envs or []):
                    try:
                        e_.close()
                    except Exception:
                        pass
                policy.model._noise_ticket = None
                continue

            best = alive[0]
            f = out / f"{suite_name}_t{tid}_ticket.npy"
            np.save(f, cands[best])
            # DOES THE WINNER ACTUALLY BEAT GAUSSIAN? Nothing checked this
            # before, and `best = alive[0]` banks a ticket unconditionally --
            # so a task where the search simply fails still produces one, and
            # using it is WORSE than not: a ticket that only matches the
            # baseline still removes the per-chunk re-draw, which this
            # benchmark prices at 25 points. object T4 ketchup is the case
            # that surfaced it: 32 survivors averaged 26.9% against a 72%
            # baseline and the best was 8/10, which a baseline-quality ticket
            # produces 44% of the time.
            base_rate = base_w[0] / max(base_r[0], 1)
            win_rate = wins[best] / max(runs[best], 1)
            p_raw = binom_sf(int(wins[best]), int(runs[best]), base_rate)

            # CERTIFICATION: the winner and Gaussian on layouts that took no
            # part in choosing it. `win_rate` is a MAXIMUM over --tickets
            # candidates and a maximum is optimistic by construction, so it
            # cannot be compared to a baseline that was not selected on. goal
            # T5/T6/T9 are what that costs: all three cleared the 15-episode
            # search check and then scored 70/75/55 on the held-out 0-19.
            # One batch covers --num_envs layouts here, because the ticket is
            # fixed and the LAYOUT varies (init_state_stride=1) -- the inverse
            # of the search, where the layout is fixed and the ticket varies.
            cert = None
            if a.certify_layouts > 0:
                c0 = a.init_state_offset + a.tiers * a.envs_per_tier
                print(f"  certifying on layouts {c0}-{c0 + a.certify_layouts - 1}"
                      f" (took no part in the selection)", flush=True)
                cw = {}
                for tag, tk in (("ticket", cands[best]), ("gaussian", None)):
                    v = (torch.from_numpy(np.asarray(tk)).to(device)
                         if tk is not None else None)
                    policy.model._noise_ticket = v if a.cycle == 1 else None
                    policy.model._noise_cycle = v if a.cycle > 1 else None
                    policy.model._noise_cycle_k = 0
                    ok = ep = 0
                    per = []
                    for g0 in range(0, a.certify_layouts, n_par):
                        n_lay = min(n_par, a.certify_layouts - g0)
                        sink = (contextlib.nullcontext() if a.verbose
                                else contextlib.redirect_stdout(io.StringIO()))
                        with sink:
                            n_ok, n_ep, _, _, _, e_ok = ev.eval_task(
                                policy, pre, post, suite, suite_name, tid,
                                n_lay, n_lay, device, a.max_episode_steps,
                                a.seed, cams, envs=envs[:n_lay],
                                init_state_offset=c0 + g0,
                                init_state_stride=1)
                        ok += n_ok; ep += n_ep; per += [int(x) for x in e_ok]
                    cw[tag] = (ok, ep, per)
                    print(f"    {tag:<8} {ok}/{ep} = {ok / max(ep, 1):.0%}",
                          flush=True)
                policy.model._noise_ticket = None
                policy.model._noise_cycle = None
                tk_r = cw["ticket"][0] / max(cw["ticket"][1], 1)
                gs_r = cw["gaussian"][0] / max(cw["gaussian"][1], 1)
                # PER LAYOUT, not just the totals. Certification is the only
                # unbiased number the search produces and both arms run the
                # SAME layouts in the same order, so it is a paired sample --
                # and storing two counts threw the pairing away. long T0's
                # 11/15 against 9/15 is p=0.43 unpaired; McNemar on the
                # discordant layouts is the sharper test, and it also names
                # which layouts the ticket wins and loses.
                cert = {"layouts": f"{c0}-{c0 + a.certify_layouts - 1}",
                        "ticket_per_layout": cw["ticket"][2],
                        "gaussian_per_layout": cw["gaussian"][2],
                        "ticket": f"{cw['ticket'][0]}/{cw['ticket'][1]}",
                        "gaussian": f"{cw['gaussian'][0]}/{cw['gaussian'][1]}",
                        "ticket_rate": round(tk_r, 4),
                        "gaussian_rate": round(gs_r, 4),
                        "passed": bool(tk_r >= gs_r + a.certify_min_margin)}
            # THREE-VALUED, because "not significant" and "not better" are
            # different and only one of them is a reason to discard the ticket.
            # 15 episodes cannot detect an improvement over a 90% baseline, so
            # a two-valued test would mark object T9's 15/15 weak -- while its
            # REPORTABLE baseline is 70% and the headroom is real. The search
            # baseline and the eval headroom are measured on different layouts
            # and routinely disagree by 20 points.
            if cert is not None and not cert["passed"]:
                # Overrides every verdict below. The search number said this
                # ticket was better; unselected layouts say it is not, and
                # those are the ones that resemble the eval.
                beats, verdict = False, (
                    f"FAILED CERTIFICATION on layouts {cert['layouts']}: "
                    f"{cert['ticket']} against Gaussian {cert['gaussian']}. "
                    f"The search rate {win_rate:.0%} was a maximum over "
                    f"{a.tickets} candidates and did not survive contact with "
                    f"layouts it was not chosen on. Banked as weak; eval will "
                    f"use Gaussian")
            elif win_rate >= 1.0 and base_rate >= 1.0:
                # A TIE AT THE CEILING IS THE ONE TIE WORTH TAKING. Both score
                # every episode, but the ticket does so DETERMINISTICALLY: its
                # result does not depend on the noise stream, while Gaussian's
                # 20/20 is one draw and a --policy_seed change flips 20.5% of
                # episodes on this benchmark. Same number, and one of them
                # survives someone re-running it. Everywhere below the ceiling
                # a tie is still a loss, because the ticket also gives up the
                # per-chunk re-draw.
                beats, verdict = None, ("ties Gaussian at 100% -- banked "
                                        "anyway: identical success, but "
                                        "deterministic rather than one draw "
                                        "from a stochastic policy")
            elif win_rate <= base_rate:
                beats, verdict = False, ("NOT BETTER than Gaussian -- eval will "
                                         "fall back to Gaussian for this task")
            elif p_raw < 0.05:
                beats, verdict = True, "BEATS the baseline"
            else:
                beats, verdict = None, ("better but not significantly; 15 "
                                        "episodes cannot resolve this against "
                                        "a high baseline. Banked, and eval "
                                        "will use it and say so -- the 20-episode "
                                        "held-out run is the real test")
            print(f"  winner {win_rate:.0%} vs Gaussian {base_rate:.0%}   "
                  f"one-sided p={p_raw:.4f} (uncorrected; {a.tickets} candidates "
                  f"searched)  ->  {verdict}", flush=True)
            meta = {
                "task": desc, "ticket_index": int(best),
                "beats_baseline": beats,
                "baseline_rate": round(base_rate, 4),
                "p_vs_baseline": round(p_raw, 5),
                "search_success": f"{wins[best]:.0f}/{runs[best]:.0f}",
                "search_rate": float(wins[best] / max(runs[best], 1)),
                "baseline_search": f"{base_w[0]:.0f}/{base_r[0]:.0f}",
                "certification": cert,
                "tickets": a.tickets, "tiers": a.tiers, "cycle": a.cycle,
                "envs_per_tier": a.envs_per_tier,
                "init_state_offset": a.init_state_offset,
                **infcfg,
                "checkpoint": str(a.checkpoint), "horizon": H, "action_dim": D,
            }
            # WOULD THIS RETIRE A WORKING TICKET? Only asked when the new
            # search is writing into a bundle that already holds one for this
            # task, which is what --overwrite against the live directory does.
            # The old ticket's standing comes from an eval that has already
            # been paid for; the new one's search rate is a maximum over
            # candidates on different layouts, so the two numbers cannot be
            # compared and the safe default is to keep what is proven.
            if not a.allow_downgrade:
                try:
                    _ex_t, _ex_m = tb.load_bundle(out)
                    _ex = _ex_m.get(tb.key(suite_name, tid), {}) \
                        if tb.key(suite_name, tid) in _ex_t else None
                except FileNotFoundError:
                    _ex = None
                if _ex is not None and _ex.get("beats_baseline") is not False:
                    print(f"  REFUSING to replace the ticket already in "
                          f"{out}: it is marked beats_baseline "
                          f"{_ex.get('beats_baseline')} with search "
                          f"{_ex.get('search_success')}, and its standing "
                          f"comes from an eval that has been run. The new "
                          f"ticket is at {out / f'{suite_name}_t{tid}_ticket.npy'}"
                          f".\n  Search into a separate --out and compare, or "
                          f"pass --allow_downgrade if you mean to retire it.",
                          flush=True)
                    results[f"{suite_name}_t{tid}"] = {
                        "task": desc, "not_banked": "would replace a working "
                        "ticket; --allow_downgrade to force",
                        "search_success": f"{wins[best]:.0f}/{runs[best]:.0f}",
                        "minutes": round((time.time() - t0) / 60, 1)}
                    (out / "search_summary.json").write_text(
                        json.dumps(results, indent=1))
                    prog.replace(prog.with_name(
                        prog.name.replace('_progress_', '_done_')))
                    for e_ in (envs or []):
                        try:
                            e_.close()
                        except Exception:
                            pass
                    policy.model._noise_ticket = None
                    continue
            # Banked the moment the task finishes, before the next one starts.
            bf = tb.save_ticket(out, suite_name, tid, cands[best], meta)
            # KEPT, not deleted. The bundle stores one ticket per task, but
            # the runner-up VECTORS exist only here -- scores_*.json records
            # every candidate's wins/runs and none of the noise. Discarding
            # them forecloses top-k sampling, which the paper (D.6.2) shows
            # performs as well as a single ticket while restoring
            # stochasticity. That matters more here than in the paper: one
            # fixed ticket makes this policy fully deterministic, and the
            # per-chunk re-draw it removes is worth 25 points by this
            # project's own measurement. 64 x 64 x 7 float32 is 115 KB.
            done_f = prog.with_name(prog.name.replace('_progress_', '_done_'))
            # THE CANDIDATE POOL IS THE ASSET, not the banked ticket. A rerun
            # in the same directory renames its progress over the old _done_
            # file and takes 64 vectors with it -- including the runner-ups
            # try_runners mines for a clean sweep of the reported layouts,
            # which is the cheapest route to a deterministic 100% there is.
            # Unlike the ticket, this cannot be recovered from anything else.
            if done_f.exists():
                n_ = 1
                while done_f.with_name(f"{done_f.stem}_prev{n_}.npz").exists():
                    n_ += 1
                keep = done_f.with_name(f"{done_f.stem}_prev{n_}.npz")
                done_f.replace(keep)
                print(f"  kept the previous candidate pool as {keep.name}",
                      flush=True)
            prog.replace(done_f)
            results[f"{suite_name}_t{tid}"] = {
                "file": str(f), "bundle": str(bf), "task": desc,
                "ticket_index": int(best),
                "search_success": f"{wins[best]:.0f}/{runs[best]:.0f}",
                "search_rate": float(wins[best] / max(runs[best], 1)),
                "baseline_search_rate": float(base_w[0] / max(base_r[0], 1)),
                "baseline_search": f"{base_w[0]:.0f}/{base_r[0]:.0f}",
                "tickets": a.tickets, "tiers": a.tiers,
                "envs_per_tier": a.envs_per_tier,
                "init_state_offset": a.init_state_offset,
                "minutes": round((time.time() - t0) / 60, 1),
            }
            (out / "search_summary.json").write_text(json.dumps(results, indent=1))
            print(f"  -> {f}  search {wins[best]:.0f}/{runs[best]:.0f} vs "
                  f"Gaussian {base_w[0]:.0f}/{base_r[0]:.0f}  "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            policy.model._noise_ticket = None
            for e_ in (envs or []):
                try:
                    e_.close()
                except Exception:
                    pass

    print(f"\nbundle: {out / tb.BUNDLE}   (+ {tb.META})")
    print("Upload both next to the checkpoint so eval can find them by task id:")
    print(f"  huggingface-cli upload <repo> {out / tb.BUNDLE} {tb.BUNDLE}")
    print(f"  huggingface-cli upload <repo> {out / tb.META} {tb.META}")
    print("The search rate is NOT the result -- it is the number the ticket was "
          "selected on, and selecting on it is what makes it optimistic. Report "
          "the ticket with eval_libero.py at --init_state_offset 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
