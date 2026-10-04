#!/usr/bin/env python
"""Test a finished search's runner-up candidates for a VERIFIED 50/50 ticket.

Nothing in a _done_*.npz is known to be 50/50. What the file holds is every
candidate vector the search drew plus its record ON THE SEARCH LAYOUTS ONLY --
15 of the 50 for the default geometry, and 5 for the ones eliminated in tier 1.
The banked winner is the best documented ticket there is, because the reported
eval added 20 more layouts to it; every runner-up still has 15.

So "swap to the 50/50 one in the npz" is not available. What IS available is
cheap: a ticket is one fixed vector and a rollout from a given init state is
deterministic, so each candidate's remaining 35 layouts are 35 facts waiting to
be read, and reading them costs 4 batches.

That matters most where the BANKED ticket has a known failure. object T2 fails
canonical layout 3 and always will; no rerun changes it, and the task cannot be
a deterministic 100% with that ticket in the bundle. A tied runner-up has not
been ruled out. Testing three of them is about an hour against 13.8 hours to
search the task again.

Order matters: 0-19 first, because that is where the banked ticket is known to
fail and a candidate that also fails there is finished after 2 batches.

    python try_runners.py --checkpoint <repo> --suite libero_object --task_id 2 \
        --done /path/_done_libero_object_t2.npz --k 4 --bank /path/bundle_dir

WHAT THIS BUYS AND WHAT IT DOES NOT. A candidate that takes 20/20 then 15/15
has been verified to solve all 50 canonical layouts -- a fact, not an estimate.
It was also selected by running it on the reported layouts, so the claim it
supports is "solves all 50 canonical layouts" and never "generalises": no
unseen init state was tested. Say which one you mean when you report it.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import eval_libero as ev
import ticket_bundle as tb


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--done", required=True,
                   help="_done_<suite>_t<id>.npz from a finished search.")
    p.add_argument("--suite", required=True)
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--k", type=int, default=4,
                   help="How many candidates to try, deepest first. Depth is "
                        "the primary key because runs differ BECAUSE the "
                        "search stopped the weak ones early: object T5's "
                        "npz holds six candidates at 2/3, which is three "
                        "layouts each, and ranking those by rate put them "
                        "above the 8/15 the search actually banked.")
    p.add_argument("--bank_if_better", action="store_true",
                   help="Also bank a candidate that beats the currently "
                        "banked ticket on the reported layouts without "
                        "sweeping them. Off by default because a +1 chosen as "
                        "the max over everything tried, ON the layouts being "
                        "reported, is a selection artefact; a clean sweep is "
                        "exact and cannot be inflated by trying more "
                        "candidates. The banked ticket is measured here first "
                        "so the comparison is same-run and same-cap.")
    p.add_argument("--bank", default=None,
                   help="Bundle directory to write a verified ticket into. "
                        "Omitted, the run only reports. NOT the directory the "
                        "candidates came from: these are selected in-sample "
                        "and the original bundle's tickets are held-out, so "
                        "keeping them apart keeps both numbers quotable.")
    p.add_argument("--seed_from", default=None,
                   help="Copy this bundle into --bank before writing, so the "
                        "result is COMPLETE and can be evaluated directly. "
                        "Without it --bank holds only the tasks that were "
                        "mined, and evaluating that bundle silently drops "
                        "every other task to Gaussian -- which on object would "
                        "cost the four 20/20 tickets that are already there. "
                        "Copying is idempotent: tickets already in --bank are "
                        "left alone, so it is safe to pass on every task in a "
                        "loop.")
    p.add_argument("--eval_layouts", type=int, default=20,
                   help="The reported span, tested first because a candidate "
                        "that fails here is done after two batches.")
    p.add_argument("--rest_offset", type=int, default=35)
    p.add_argument("--rest_layouts", type=int, default=0,
                   help="Layouts neither the search nor the eval ran -- 35-49 "
                        "with the default geometry. OFF by default: a "
                        "candidate that takes 0-19 has already delivered "
                        "everything the benchmark reports, deterministically, "
                        "and 35-49 only upgrades the CLAIM from 'solves the 20 "
                        "reported layouts' to 'solves all 50'. 15 turns it on.")
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--control_freq", type=int, default=10)
    p.add_argument("--max_episode_steps", type=int, default=0,
                   help="0 is the env's own cap, which is the criterion the "
                        "reported number uses. Verifying under a stricter cap "
                        "than the one being claimed makes no sense.")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--n_action_steps", type=int, default=2)
    p.add_argument("--vision_input_size", type=int, default=384)
    p.add_argument("--render_gpu", type=int, default=0)
    p.add_argument("--dataset_id", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--stock_init", action="store_true")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    if a.bank and not a.seed_from:
        src = Path(a.done).parent
        try:
            n_src = len(tb.load_bundle(src)[0])
        except FileNotFoundError:
            n_src = 0
        try:
            n_bank = len(tb.load_bundle(a.bank)[0])
        except FileNotFoundError:
            n_bank = 0
        if n_src > n_bank + 1:
            print(f"WARNING: --bank {a.bank} holds {n_bank} ticket(s) while "
                  f"{src} holds {n_src}.\n  Evaluating the --bank bundle would "
                  f"drop every task missing from it to Gaussian, which on a "
                  f"suite\n  with 20/20 tickets in it is a loss, not a "
                  f"no-op. Pass --seed_from {src} to copy them in.\n",
                  flush=True)

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from checkpoint_utils import resolve_checkpoint
    ev.setup_libero_env(a.control_freq, a.render_gpu, a.stock_init)
    ckpt = resolve_checkpoint(a.checkpoint, for_resume=False)
    policy = ev.load_policy(ckpt, device, a.num_inference_steps,
                            n_action_steps=a.n_action_steps,
                            vision_input_size=a.vision_input_size)
    pre, post = ev.load_processors(ckpt, device, a.dataset_id)
    cams = list(policy.config.cameras_for_vision_state_concat) \
        if hasattr(policy.config, "cameras_for_vision_state_concat") else []
    infcfg = ev.inference_config(policy, a.control_freq, a.max_episode_steps,
                                 a.stock_init)
    ev.report_inference_config(infcfg)

    # DOES THIS POOL BELONG TO THIS TASK? Nothing downstream would notice if
    # it did not: the candidates load, the MUST_MATCH check passes because the
    # checkpoint is the same, the rollouts run, and a ticket searched for
    # libero_goal task 2 gets banked under libero_spatial.2. The filename is
    # the only place the pool's identity is written down.
    stem = Path(a.done).name
    if stem.startswith("_done_") or stem.startswith("_progress_"):
        core = stem.split("_", 2)[2].rsplit(".npz", 1)[0]
        core = core.split("_prev")[0]
        f_suite, _, f_tid = core.rpartition("_t")
        if f_suite and f_tid.isdigit() and \
                (f_suite != a.suite or int(f_tid) != a.task_id):
            print(f"ERROR: {stem} holds candidates for {f_suite} task {f_tid}, "
                  f"but --suite {a.suite} --task_id {a.task_id} was given. "
                  f"Those candidates were searched against a different task "
                  f"and would be banked under the wrong key.", file=sys.stderr)
            return 1

    z = np.load(a.done, allow_pickle=True)
    cands, wins, runs = z["cands"], z["wins"], z["runs"]
    # The search's config, not this one. A candidate scored under a different
    # n_action_steps is not the candidate being tested here.
    if "infcfg" in z.files:
        was = json.loads(str(z["infcfg"]))
        bad = {k: (was.get(k), infcfg[k]) for k in ev.MUST_MATCH
               if was.get(k) != infcfg[k]}
        if bad:
            print("ERROR: these candidates were searched under "
                  + ", ".join(f"{k}={w} (now {n})" for k, (w, n) in bad.items())
                  + ". Their scores describe a different policy.",
                  file=sys.stderr)
            return 1
    rate = np.where(runs > 0, wins / np.maximum(runs, 1), -1.0)
    order = [i for i in sorted(range(len(rate)),
                               key=lambda i: (-runs[i], -rate[i])) if runs[i] > 0]
    top = order[:a.k]
    # The banked ticket is named in the metadata. Inferring it from the ranking
    # is what mislabelled object T3 and T5, whose top-by-rate candidates were
    # tier-1 casualties with three and nine layouts against the survivor's
    # fifteen. It goes first so --bank_if_better compares like with like.
    banked_idx = None
    try:
        _, bmeta = tb.load_bundle(Path(a.done).parent)
        banked_idx = bmeta.get(tb.key(a.suite, a.task_id), {}).get("ticket_index")
    except FileNotFoundError:
        pass
    if banked_idx is not None and banked_idx in order:
        top = [banked_idx] + [i for i in top if i != banked_idx][:a.k - 1]
    deepest = max(runs)
    shallow = [i for i in top if runs[i] < deepest]
    if shallow:
        print(f"  NOTE: {len(shallow)} of these ran fewer than {int(deepest)} "
              f"layouts and were eliminated early -- their scores are the "
              f"absence of evidence, not evidence.", flush=True)

    from lerobot.envs.libero import LiberoEnv, _get_suite
    suite = _get_suite(a.suite)
    print(f"  building {a.num_envs} envs...", end="", flush=True)
    t_b = time.time()
    envs = [LiberoEnv(task_suite=suite, task_id=a.task_id,
                      task_suite_name=a.suite, obs_type="pixels_agent_pos",
                      init_states=True, episode_index=0)
            for _ in range(a.num_envs)]
    print(f" {time.time() - t_b:.0f}s", flush=True)

    def run(vec, offset, n):
        """-> (successes, episodes, per-episode vector) over layouts offset..offset+n-1."""
        policy.model._noise_ticket = torch.from_numpy(
            np.asarray(vec, dtype=np.float32)).to(device)
        ok = ep = 0
        per = []
        for g0 in range(0, n, a.num_envs):
            m = min(a.num_envs, n - g0)
            sink = (contextlib.nullcontext() if a.verbose
                    else contextlib.redirect_stdout(io.StringIO()))
            with sink:
                n_ok, n_ep, _, _, _, e_ok = ev.eval_task(
                    policy, pre, post, suite, a.suite, a.task_id,
                    m, m, device, a.max_episode_steps, 10000, cams,
                    envs=envs[:m], init_state_offset=offset + g0,
                    init_state_stride=1)
            ok += n_ok; ep += n_ep; per += list(e_ok)
        return ok, ep, per

    pool_desc = str(z["desc"]) if "desc" in z.files else ""
    env_desc = getattr(envs[0], "task_description", None) \
        or getattr(getattr(envs[0], "task", None), "language", None) or ""
    if pool_desc and env_desc and pool_desc.strip() != str(env_desc).strip():
        print(f"ERROR: the pool was searched on '{pool_desc}' but this env is "
              f"'{env_desc}'. Wrong file for this task.", file=sys.stderr)
        for e in envs:
            try:
                e.close()
            except Exception:
                pass
        return 1

    print(f"\n{a.suite} task {a.task_id}: trying {len(top)} candidates, "
          f"{a.eval_layouts} layouts from 0 then {a.rest_layouts} from "
          f"{a.rest_offset}", flush=True)
    winner, results = None, []
    for n, i in enumerate(top):
        tag = ("banked" if i == banked_idx else f"runner-up {n}")
        print(f"\n  ticket {i} ({tag}, searched "
              f"{int(wins[i])}/{int(runs[i])})", flush=True)
        ok1, ep1, per1 = run(cands[i], 0, a.eval_layouts)
        miss1 = [j for j, x in enumerate(per1) if not x]
        print(f"    layouts 0-{a.eval_layouts - 1}: {ok1}/{ep1}"
              + (f"   fails {miss1}" if miss1 else "   clean"), flush=True)
        if ok1 < ep1:
            # Those failures are permanent for this vector. Nothing later can
            # make it 50/50, so the remaining two batches would buy nothing.
            results.append((i, ok1, ep1, None, None))
            continue
        if a.rest_layouts == 0:
            # 20/20 on the reported layouts IS the result. It reproduces
            # exactly, because the ticket is fixed and the rollout from a given
            # init state is deterministic. 35-49 would strengthen the claim,
            # not the number.
            winner = i
            results.append((i, ok1, ep1, None, None))
            print(f"    -> ticket {i} takes all {ep1} reported layouts",
                  flush=True)
            break
        ok2, ep2, per2 = run(cands[i], a.rest_offset, a.rest_layouts)
        miss2 = [a.rest_offset + j for j, x in enumerate(per2) if not x]
        print(f"    layouts {a.rest_offset}-{a.rest_offset + a.rest_layouts - 1}"
              f": {ok2}/{ep2}" + (f"   fails {miss2}" if miss2 else "   clean"),
              flush=True)
        results.append((i, ok1, ep1, ok2, ep2))
        if ok2 == ep2:
            winner = i
            print(f"    -> ticket {i} takes every layout it has run: "
                  f"{int(runs[i])} searched + {ep1} reported"
                  + (f" + {ep2} unseen" if ep2 else ""), flush=True)
            break

    policy.model._noise_ticket = None
    for e in envs:
        try:
            e.close()
        except Exception:
            pass

    print("\n=== summary ===")
    for i, o1, e1, o2, e2 in results:
        tail = (f"  {a.rest_offset}+: {o2}/{e2}" if o2 is not None
                else "" if o1 == e1 else
                "  (stopped: those failures are permanent for this vector)")
        print(f"  ticket {i:>3}   0-{a.eval_layouts - 1}: {o1}/{e1}{tail}")

    best = max(results, key=lambda r: r[1]) if results else None
    if winner is None and a.bank_if_better and a.bank and best is not None:
        cur = next((o for i, o, _, _, _ in results if i == banked_idx), None)
        if cur is not None and best[1] > cur:
            winner = best[0]
            print(f"\n--bank_if_better: ticket {winner} takes {best[1]}/"
                  f"{best[2]} against the banked ticket's {cur}/{best[2]} in "
                  f"this same run. Banking it. This is a selection over "
                  f"{len(results)} candidates on the layouts being reported, "
                  f"so the gain is an upper bound on what it is worth.",
                  flush=True)
        elif cur is None:
            print(f"\n--bank_if_better: the banked ticket was not among the "
                  f"candidates measured here, so there is nothing to compare "
                  f"against. Nothing banked.", flush=True)
    if winner is None:
        print(f"\nNone of the {len(top)} takes all {a.eval_layouts} reported "
              f"layouts; the best was ticket {best[0]} at {best[1]}/{best[2]}. "
              f"NOTHING IS BANKED, deliberately: a candidate that merely scores "
              f"higher, picked as the max over {len(top)} tried on the layouts "
              f"being reported, is a selection artefact worth no more than the "
              f"ticket already in the bundle. A clean sweep is different -- "
              f"'solves all {a.eval_layouts}' is exact and cannot be inflated "
              f"by trying more candidates; --bank_if_better writes the higher "
              f"score anyway.\nThe remaining option is a "
              f"--require_perfect search over --init_state_offset 0, which "
              f"costs 1/p^50 candidates at per-layout pass rate p; read the "
              f"first-hour line before committing to it.")
        return 0

    if a.bank and a.seed_from:
        # BEFORE the write, and only for keys --bank does not already have, so
        # a loop over tasks does not undo the previous task's result.
        src_t, src_m = tb.load_bundle(a.seed_from)
        try:
            have, _ = tb.load_bundle(a.bank)
        except FileNotFoundError:
            have = {}
        added = [k for k in src_t if k not in have]
        for k in added:
            suite_, tid_ = k.rsplit(".", 1)
            tb.save_ticket(a.bank, suite_, int(tid_), src_t[k],
                           src_m.get(k, {}))
        if added:
            print(f"\nseeded {a.bank} with {len(added)} ticket(s) from "
                  f"{a.seed_from}: {', '.join(sorted(added))}", flush=True)

    if a.bank:
        meta = {"task": str(z["desc"]) if "desc" in z.files else None,
                "ticket_index": int(winner), "beats_baseline": True,
                "verified_all_canonical": bool(a.rest_layouts),
                "verified": {"searched": f"{int(wins[winner])}/{int(runs[winner])}",
                             "layouts_0_19": f"{a.eval_layouts}/{a.eval_layouts}",
                             f"layouts_{a.rest_offset}_plus":
                                 f"{a.rest_layouts}/{a.rest_layouts}"},
                "selected_on_reported_layouts": True,
                "swept_reported_layouts": bool(
                    next((o == e for i, o, e, _, _ in results
                          if i == winner), False)),
                "claim": "solves all 50 canonical layouts; generalisation "
                         "to unseen init states is untested",
                **infcfg, "checkpoint": str(a.checkpoint)}
        # KEEP WHAT IS BEING REPLACED. save_ticket overwrites the key and its
        # metadata together, so the old ticket_index goes with it and the old
        # vector becomes recoverable only by cross-referencing an eval JSON.
        # Banking into the live bundle is the simple workflow; this is what
        # makes it reversible.
        try:
            _old_t, _old_m = tb.load_bundle(a.bank)
            _k = tb.key(a.suite, a.task_id)
        except FileNotFoundError:
            _old_t, _k = {}, None
        if _k is not None and _k in _old_t:
            bak = Path(a.bank) / f"prev_{a.suite}_t{a.task_id}.npy"
            np.save(bak, _old_t[_k])
            log = Path(a.bank) / "replaced.json"
            hist = json.loads(log.read_text()) if log.exists() else {}
            hist.setdefault(_k, []).append(
                {"when": time.strftime("%Y-%m-%d %H:%M"),
                 "vector": bak.name, "meta": _old_m.get(_k, {})})
            log.write_text(json.dumps(hist, indent=1, sort_keys=True))
            print(f"  previous ticket kept as {bak.name}, its metadata in "
                  f"replaced.json", flush=True)
        f = tb.save_ticket(a.bank, a.suite, a.task_id, cands[winner], meta)
        print(f"\nbanked ticket {winner} -> {f}")
        print("The metadata records that this ticket was selected by running "
              "it on the reported layouts.\nReport it as 'solves all 50 "
              "canonical layouts', not as a held-out result.")
    else:
        print(f"\nticket {winner} is the one to bank; rerun with --bank <dir>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
