"""Kill the ticket search at every point and prove the resume is identical.

No GPU, no LIBERO: the rollouts are a lookup table, so what is under test is
purely the bookkeeping -- next_tier, done_k, base_done, alive -- which is
where every bug in this file has been. Four of them shipped:

  * score_baseline wrote done_k = envs_per_tier, so a death between the
    baseline and the candidates made tier 1 score nothing (327853f)
  * resuming across a --max_episode_steps change added episodes at two
    different caps to one numerator (10be64d)
  * resuming under a different --envs_per_tier re-ran layouts an earlier
    tier had already scored (f33c201)
  * the baseline was indexed by tier, so a resume at tier > 0 measured the
    Gaussian on layouts no candidate would ever run

Each was invisible until hours of GPU time had been spent. Run with
`python test_search_resume.py`.
"""
from math import ceil
import random

M, TIERS, N, NUM_ENVS = 5, 3, 64, 10


class Killed(Exception):
    pass


def search(outcome, base_lay, kill_at=None, state=None):
    """One task. `kill_at` raises after that many eval_task BATCHES.

    `state` is the progress file: the exact fields the real _save writes.
    Returns (winner, wins, runs, calls) or raises Killed with state updated
    in place, which is what a Colab cutoff does.
    """
    calls = [0]

    def rollout(kind, k, group):
        calls[0] += 1
        if kill_at is not None and calls[0] > kill_at:
            raise Killed
        if kind == "base":
            return base_lay[k]
        return [outcome[(i, k)] for i in group]

    st = state
    wins, runs = st["wins"], st["runs"]
    alive, first_tier = st["alive"], st["next_tier"]
    start_k0, base_k = st["done_k"], st["base_k"]
    base_lw, base_lr = st["base_lw"], st["base_lr"]

    def save(tier, done_k, alive_now):
        st.update(next_tier=tier, done_k=done_k, alive=list(alive_now),
                  base_k=base_k[0], wins=list(wins), runs=list(runs),
                  base_lw=list(base_lw), base_lr=list(base_lr))

    base_k = [base_k]

    # --- the self-heal for pre-327853f files -----------------------------
    if start_k0 > 0 and sum(runs) == 0:
        start_k0 = 0

    def floor_for(tier):
        lo, hi = tier * M, (tier + 1) * M
        seg = sum(base_lr[lo:hi])
        if seg > 0:
            return sum(base_lw[lo:hi]) / seg
        return sum(base_lw) / max(sum(base_lr), 1)

    def score_baseline(tier, cand_k):
        if base_k[0] >= TIERS * M:
            return
        for k in range(base_k[0], TIERS * M):   # NOT offset by tier
            w = rollout("base", k, None)
            base_lw[k] += w
            base_lr[k] += NUM_ENVS
            base_k[0] = k + 1
            save(tier, cand_k, alive)        # the CANDIDATES' count, not ours

    def score(idx_list, tier, start_k):
        floor, n_end = floor_for(tier), (tier + 1) * M
        idx_list = list(idx_list)
        for k in range(start_k, M):
            layout = tier * M + k
            for g0 in range(0, len(idx_list), NUM_ENVS):
                grp = idx_list[g0:g0 + NUM_ENVS]
                ep = rollout("cand", layout, grp)   # ONE call = one batch
                for j, i in enumerate(grp):
                    wins[i] += ep[j]
                    runs[i] += 1
            if floor > 0 and k + 1 < M and tier < TIERS - 1:
                rem = M - (k + 1)
                live = [i for i in idx_list
                        if (wins[i] + rem) / n_end >= floor - 1e-9]
                idx_list = live or [max(idx_list, key=lambda i: wins[i])]
            save(tier, k + 1, idx_list)
        return idx_list

    for tier in range(first_tier, TIERS):
        if tier == 0:
            score_baseline(tier, start_k0 if tier == first_tier else 0)
        alive = score(alive, tier, start_k0 if tier == first_tier else 0)
        rate = [wins[i] / runs[i] if runs[i] else -1.0 for i in range(N)]
        alive = sorted(alive, key=lambda i: -rate[i])
        if tier < TIERS - 1:
            floor = floor_for(tier)
            kept = [i for i in alive if runs[i] and wins[i] / runs[i] >= floor - 1e-9]
            alive = kept or alive[:1]
        assert runs[alive[0]] > 0, "tier scored nothing"
        save(tier + 1, 0, alive)
    return alive[0], list(wins), list(runs), calls[0]


def fresh():
    return {"wins": [0] * N, "runs": [0] * N, "alive": list(range(N)),
            "next_tier": 0, "done_k": 0, "base_k": 0,
            "base_lw": [0.0] * (TIERS * M), "base_lr": [0.0] * (TIERS * M)}


def one_seed(seed, baseline):
    rng = random.Random(seed)
    truth = [rng.betavariate(2, 2.4) for _ in range(N)]
    truth[7] = 1.0
    outcome = {(i, L): int(rng.random() < truth[i])
               for i in range(N) for L in range(TIERS * M)}
    base_lay = [sum(rng.random() < baseline for _ in range(NUM_ENVS))
                for _ in range(TIERS * M)]

    ref = search(outcome, base_lay, state=fresh())
    total = ref[3]

    # A lease is how many batches one Colab session gets. It has to cover the
    # widest gap between two saves -- one layout of tier 1, ceil(N/NUM_ENVS)
    # batches -- or no lease can ever finish a layout and the search cannot
    # advance no matter how many times it is restarted. That bound is the
    # checkpoint granularity, and it is worth knowing: at 6 min a batch it is
    # 42 minutes of work at risk.
    min_lease = ceil(N / NUM_ENVS)
    for lease in range(min_lease, total + 1):
        st, deaths = fresh(), 0
        while True:                       # die repeatedly, like a real Colab
            try:
                got = search(outcome, base_lay, kill_at=lease, state=st)
                break
            except Killed:
                deaths += 1
                assert deaths <= total, (
                    f"lease {lease} makes no progress: a save-to-save gap is "
                    f"wider than the lease")
        if got[:3] != ref[:3]:
            return (f"seed {seed} baseline {baseline:.0%}: a {lease}-batch "
                    f"lease (of {total}) changed the result\n"
                    f"  fresh   winner {ref[0]} {ref[1][ref[0]]}/{ref[2][ref[0]]}\n"
                    f"  resumed winner {got[0]} {got[1][got[0]]}/{got[2][got[0]]}")
    return None


def main():
    bad = 0
    for baseline in (0.96, 0.80, 0.60, 0.40):
        for seed in range(6):
            err = one_seed(seed, baseline)
            if err:
                print("FAIL " + err)
                bad += 1
        print(f"baseline {baseline:.0%}: every kill point resumes identically"
              if not bad else f"baseline {baseline:.0%}: {bad} failures")
    print("\nOK" if not bad else f"\n{bad} FAILURES")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
