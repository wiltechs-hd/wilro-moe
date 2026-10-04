#!/usr/bin/env python
"""Look at the tickets a search selected, and at what separated them from the pool.

    python plot_tickets.py --bundle /path/to/goal_tickets --out tickets.png
    python plot_tickets.py --bundle ... --key libero_object.5 --out t5.png

The four panels answer four different questions, and three of them need the
_done_*.npz pools that sit beside the bundle, because a single ticket is one
draw from N(0, I) and nothing about it is interpretable on its own. What is
interpretable is how the winners differ from the candidates that lost.

  A  the winner as a horizon x action-dim field. The honest expectation is
     that it looks like noise; "it looks like noise" is a result, not a
     failure, and it is what rules out a simple story like "the ticket is a
     constant offset on the gripper channel".

  B  norm against score, every candidate in the pool. A draw of 64 x 7 from
     N(0, I) has expected norm sqrt(448) = 21.2, and if winners sat
     systematically off that line the effect would be SCALE, not direction --
     which would make the whole search replaceable by one scalar.

  C  separation per horizon position. Only the first --n_action_steps rows of
     each chunk are ever executed; the rest shape the trajectory only through
     the flow ODE. If the winners differ from the losers mainly in the
     executed rows, the ticket is doing something local and legible. If the
     separation is flat across all 64, it is not.

  D  cosine similarity between the banked tickets. Near zero everywhere means
     every task found its own direction in a 448-dim space, which is what
     random high-dimensional vectors do and what a per-task search implies.
     A block of high similarity would mean a shared direction exists and one
     ticket might serve several tasks.

Colors: panels A and D are signed, so they use a diverging blue-red ramp with a
neutral gray midpoint -- never a rainbow, because a rainbow invents an ordering
the data does not have. B and C are single-series and carry no legend.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The reference palette's diverging pair and first two categorical slots.
BLUE, RED, GRAY = "#2a78d6", "#d03b3b", "#f0efec"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bwr_", [BLUE, GRAY, RED])


def load(bundle: Path):
    """-> (tickets, meta, {key: done_npz_path})."""
    import ticket_bundle as tb
    t, m = tb.load_bundle(bundle)
    d = Path(bundle)
    d = d.parent if d.is_file() else d
    pools = {}
    for k in t:
        suite, tid = k.rsplit(".", 1)
        f = d / f"_done_{suite}_t{tid}.npz"
        if f.exists():
            pools[k] = f
    return t, m, pools


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--key", default=None,
                   help="Which ticket to show in panels A-C. Default: the one "
                        "with the highest search rate that has a pool.")
    p.add_argument("--n_action_steps", type=int, default=2,
                   help="Rows of each chunk that are actually executed; panel "
                        "C marks them.")
    p.add_argument("--top_frac", type=float, default=0.25)
    p.add_argument("--out", default="tickets.png")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tickets, meta, pools = load(Path(a.bundle))
    if not tickets:
        print("no tickets in that bundle")
        return 1

    key = a.key
    if key is None:
        cand = [k for k in tickets if k in pools] or list(tickets)
        key = max(cand, key=lambda k: meta.get(k, {}).get("search_rate", 0.0))
    if key not in tickets:
        print(f"{key} not in bundle; have {sorted(tickets)}")
        return 1
    vec = np.asarray(tickets[key], dtype=float)
    cyc = vec.ndim == 3
    shown = vec[0] if cyc else vec            # (H, D)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=SURFACE)
    for row in ax:
        for x in row:
            x.set_facecolor(SURFACE)
            for s in ("top", "right"):
                x.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                x.spines[s].set_color("#d8d7d2")
            x.tick_params(colors=INK_2, labelsize=8, length=3)

    # --- A: the winner itself -------------------------------------------
    A = ax[0][0]
    lim = float(np.abs(shown).max())
    im = A.imshow(shown.T, aspect="auto", cmap=diverging(),
                  vmin=-lim, vmax=lim, interpolation="nearest")
    A.set_title(f"A  {key}" + (f", cycle vector 1 of {vec.shape[0]}" if cyc else "")
                + f"   |x| = {np.linalg.norm(shown):.1f}",
                color=INK, fontsize=10, loc="left", pad=8)
    A.set_xlabel("horizon step", color=INK_2, fontsize=9)
    A.set_ylabel("action dim", color=INK_2, fontsize=9)
    A.axvline(a.n_action_steps - 0.5, color=INK, lw=1.2, ls=":")
    A.text(a.n_action_steps + 0.6, -0.9, "executed", color=INK, fontsize=8)
    cb = fig.colorbar(im, ax=A, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=INK_2, labelsize=7)
    cb.outline.set_visible(False)

    pool = pools.get(key)
    # --- B: norm against score ------------------------------------------
    B = ax[0][1]
    if pool:
        z = np.load(pool, allow_pickle=True)
        c, w, r = z["cands"], z["wins"], z["runs"]
        live = r > 0
        flat = c[live].reshape(live.sum(), -1)
        norms = np.linalg.norm(flat, axis=1)
        rate = w[live] / r[live]
        B.scatter(norms, rate * 100, s=22, c=SERIES_1, alpha=0.55,
                  edgecolors="none")
        exp = np.sqrt(flat.shape[1])
        B.axvline(exp, color=INK, lw=1.2, ls=":")
        B.text(exp, B.get_ylim()[1], f"  E|x| for N(0,I) = {exp:.1f}",
               color=INK, fontsize=8, va="top")
        rho = np.corrcoef(norms, rate)[0, 1]
        B.set_title(f"B  norm vs search rate, {live.sum()} candidates   "
                    f"r = {rho:+.2f}", color=INK, fontsize=10, loc="left", pad=8)
        B.set_xlabel("||x||", color=INK_2, fontsize=9)
        B.set_ylabel("search rate (%)", color=INK_2, fontsize=9)
    else:
        B.text(.5, .5, f"no _done_ pool beside the bundle for {key}",
               ha="center", color=INK_2, fontsize=9)
        B.set_axis_off()

    # --- C: separation per horizon position ------------------------------
    C = ax[1][0]
    if pool:
        order = np.argsort(-rate)
        k = max(2, int(len(order) * a.top_frac))
        hi = flat[order[:k]].reshape(k, *shown.shape)
        lo = flat[order[-k:]].reshape(k, *shown.shape)
        # Pooled per-element effect size, averaged over action dims. Raw mean
        # differences are not comparable across positions; this is.
        sd = np.sqrt((hi.var(0) + lo.var(0)) / 2) + 1e-9
        d_pos = (np.abs(hi.mean(0) - lo.mean(0)) / sd).mean(1)
        C.plot(d_pos, color=SERIES_1, lw=2)
        C.axvspan(-0.5, a.n_action_steps - 0.5, color=SERIES_2, alpha=0.18, lw=0)
        C.text(a.n_action_steps + 0.6, C.get_ylim()[1] * 0.95,
               "executed rows", color=INK_2, fontsize=8, va="top")
        # What the same statistic reads on noise: two groups of k drawn from
        # the same distribution. Without it a flat 0.3 looks like a signal.
        rng = np.random.default_rng(0)
        null = []
        for _ in range(40):
            s = rng.permutation(len(flat))
            h2 = flat[s[:k]].reshape(k, *shown.shape)
            l2 = flat[s[k:2 * k]].reshape(k, *shown.shape)
            sd2 = np.sqrt((h2.var(0) + l2.var(0)) / 2) + 1e-9
            null.append((np.abs(h2.mean(0) - l2.mean(0)) / sd2).mean(1))
        nq = np.quantile(np.array(null), 0.95, axis=0)
        C.plot(nq, color=INK_2, lw=1.2, ls="--")
        C.text(len(nq) - 1, nq[-1], " 95th pct of a shuffled split",
               color=INK_2, fontsize=8, ha="right", va="bottom")
        C.set_title(f"C  top {k} vs bottom {k}: effect size per horizon step",
                    color=INK, fontsize=10, loc="left", pad=8)
        C.set_xlabel("horizon step", color=INK_2, fontsize=9)
        C.set_ylabel("|mean diff| / pooled sd", color=INK_2, fontsize=9)
    else:
        C.set_axis_off()

    # --- D: do the banked tickets share a direction? ---------------------
    D = ax[1][1]
    keys = sorted(tickets)
    flat_t = []
    for kk in keys:
        v = np.asarray(tickets[kk], dtype=float)
        flat_t.append((v[0] if v.ndim == 3 else v).ravel())
    M = np.array(flat_t)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = M @ M.T
    lim = max(0.2, float(np.abs(S - np.eye(len(S))).max()))
    im2 = D.imshow(S, cmap=diverging(), vmin=-lim, vmax=lim)
    D.set_xticks(range(len(keys)))
    D.set_yticks(range(len(keys)))
    short = [k.split(".")[-1] for k in keys]
    D.set_xticklabels(short, fontsize=7, color=INK_2)
    D.set_yticklabels(short, fontsize=7, color=INK_2)
    off = S[~np.eye(len(S), dtype=bool)]
    D.set_title(f"D  cosine between banked tickets   off-diagonal |max| "
                f"{np.abs(off).max():.3f}", color=INK, fontsize=10,
                loc="left", pad=8)
    cb2 = fig.colorbar(im2, ax=D, fraction=0.046, pad=0.02)
    cb2.ax.tick_params(colors=INK_2, labelsize=7)
    cb2.outline.set_visible(False)

    fig.suptitle(f"{Path(a.bundle).name}: {len(tickets)} banked tickets",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {a.out}")

    # The numbers behind the panels, so the figure does not have to be
    # squinted at to be quoted.
    print(f"\n{key}: shape {tuple(vec.shape)}, norm {np.linalg.norm(shown):.2f} "
          f"against E = {np.sqrt(shown.size):.2f} for a N(0,I) draw")
    if pool:
        print(f"  norm-vs-rate correlation over {live.sum()} candidates: "
              f"{rho:+.3f}")
        print(f"  effect size per horizon step: max {d_pos.max():.2f} at step "
              f"{int(d_pos.argmax())}, mean {d_pos.mean():.2f}; "
              f"shuffled 95th pct mean {nq.mean():.2f}")
        ex = d_pos[:a.n_action_steps].mean()
        print(f"  executed rows mean {ex:.2f} vs rest {d_pos[a.n_action_steps:].mean():.2f}")
    print(f"  largest off-diagonal cosine between banked tickets: "
          f"{np.abs(off).max():.4f} (random 448-dim vectors average "
          f"{np.sqrt(2 / (np.pi * M.shape[1])):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
