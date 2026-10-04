"""One file of golden tickets, keyed by (suite, task_id), living next to a checkpoint.

A ticket is bound to the policy that was searched with it -- same weights,
same horizon, same control frequency -- so the natural place to keep it is the
checkpoint repo itself:

    huggingface-cli upload <repo> ./tickets/golden_tickets.safetensors
    huggingface-cli upload <repo> ./tickets/golden_tickets.json

Eval then takes `--noise_tickets auto` and looks each task up by id.

PARTIAL BUNDLES ARE NORMAL AND MUST BE VISIBLE. Search is hours per task, so a
bundle will usually cover some tasks and not others. A suite average that
mixes ticketed and Gaussian tasks is not comparable to anything, so the loader
reports coverage and the eval writes it into the result JSON rather than
letting it pass silently.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

BUNDLE = "golden_tickets.safetensors"
META = "golden_tickets.json"


def key(suite: str, task_id: int) -> str:
    return f"{suite}.{int(task_id)}"


def save_ticket(out_dir, suite: str, task_id: int, ticket: np.ndarray, meta: dict):
    """Read-modify-write. Bundles are a few hundred KB; rewriting is free and
    it means a run killed mid-search still leaves every finished task on disk.

    NOT SAFE FOR CONCURRENT SEARCHES ON ONE DIRECTORY, and there is no lock.
    Two processes that both read {goal.0}, then write {goal.0, goal.1} and
    {goal.0, object.0}, leave whichever finished first erased. Give each
    concurrent search its own --out and `merge()` them at the end.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f, m = out / BUNDLE, out / META
    tensors = load_file(str(f)) if f.exists() else {}
    info = json.loads(m.read_text()) if m.exists() else {}
    k = key(suite, task_id)
    tensors[k] = np.asarray(ticket, dtype=np.float32)
    info[k] = meta
    save_file(tensors, str(f))
    m.write_text(json.dumps(info, indent=1, sort_keys=True))
    return f


def load_bundle(path):
    """-> (dict[key] -> (H, D) array, dict[key] -> meta). `path` may be the
    bundle file or a directory containing it (a checkpoint, for instance)."""
    p = Path(path)
    f = p if p.is_file() else p / BUNDLE
    if not f.exists():
        raise FileNotFoundError(
            f"no {BUNDLE} at {p}. Search produces it; if the tickets live on "
            f"the hub, they must be downloaded with the checkpoint.")
    m = f.with_name(META)
    return load_file(str(f)), (json.loads(m.read_text()) if m.exists() else {})


def coverage(tensors, suite: str, task_ids) -> dict:
    """Which of the tasks about to be evaluated actually have a ticket."""
    have = [t for t in task_ids if key(suite, t) in tensors]
    miss = [t for t in task_ids if key(suite, t) not in tensors]
    return {"with_ticket": have, "gaussian": miss,
            "n_with_ticket": len(have), "n_tasks": len(list(task_ids))}


def top_k(done_npz, k: int = 8, min_rate: float = 0.0):
    """-> (k, H, D) of the best-scoring candidates from a finished search.

    The bundle keeps one ticket per task; this recovers the rest from the
    _done_*.npz the search now leaves behind. The paper (D.6.2) finds that
    drawing uniformly from the top-k performs as well as the single best while
    restoring stochasticity -- which matters here, because one fixed ticket
    makes the policy deterministic and removes the per-chunk re-draw this
    benchmark measured at 25 points.

    Ranked on CUMULATIVE wins/runs, so candidates eliminated early (5 episodes)
    are compared against survivors (15). That favours survivors, which is the
    intent: an early exit means the evidence stopped at "not promising".
    """
    z = np.load(done_npz, allow_pickle=True)
    w, r, cands = z["wins"], z["runs"], z["cands"]
    rate = np.where(r > 0, w / np.maximum(r, 1), -1.0)
    order = sorted(range(len(rate)), key=lambda i: (-rate[i], -r[i]))
    keep = [i for i in order if rate[i] >= min_rate][:k]
    return cands[keep], [(int(i), int(w[i]), int(r[i])) for i in keep]


def report(path) -> int:
    """Print what a bundle actually contains. -> number of weak entries.

    Search banks a ticket for every task it finishes, including the ones it
    could not beat Gaussian on -- deliberately, so the candidates and the
    metadata survive -- and the search log then reports every finished task
    the same way. object T5 went in at 53% against an 85% baseline and reads
    as "already in the bundle" next to T9's 15/15.
    """
    tensors, info = load_bundle(path)
    rows, weak = [], 0
    for k in sorted(tensors):
        m = info.get(k, {})
        b = m.get("beats_baseline")
        verdict = {True: "BEATS", False: "weak -- eval falls back to Gaussian",
                   None: "banked, unresolved"}[b if b in (True, False) else None]
        if b is False:
            weak += 1
            if m.get("disabled_note"):
                verdict = ("disabled by hand -- it beat its search baseline "
                           "and lost on the reported layouts")
        c = m.get("certification")
        cert = (f"{c['ticket']} vs {c['gaussian']}" if c else "not certified")
        # PROVENANCE, because the two kinds of ticket produce numbers that look
        # identical. One was searched on layouts the eval never touches and its
        # suite average is a held-out result; the other was picked by running
        # candidates on the reported layouts, so its average is exact but
        # in-sample. A bundle holding both is fine -- the workflow wants one
        # bundle -- as long as the report says which is which.
        if m.get("selected_on_reported_layouts"):
            verdict = ("SWEPT the reported layouts (in-sample)"
                       if m.get("swept_reported_layouts")
                       else "better on the reported layouts (in-sample)")
        rows.append((k, m.get("search_success", "?"),
                     m.get("baseline_search", "?"), cert, verdict))
    w0 = max([len(r[0]) for r in rows] + [4])
    w3 = max([len(r[3]) for r in rows] + [13])
    print(f"{'task':<{w0}}  {'search':>8}  {'gaussian':>9}  "
          f"{'certified':<{w3}}  verdict")
    for k, sc, bl, ct, v in rows:
        print(f"{k:<{w0}}  {sc:>8}  {bl:>9}  {ct:<{w3}}  {v}")
    print(f"\n{len(rows)} tickets, {weak} weak.")
    if weak:
        print("The weak ones are inert at eval time -- it checks "
              "beats_baseline and uses Gaussian -- so they cost nothing to "
              "leave in place. Re-search them with more --tickets; the search "
              "skips them by default, --retry_failed redoes them.")
    print("SEARCH RATE IS NOT THE RESULT. It is the number the ticket was "
          "selected on. Report with eval_libero.py --init_state_offset 0.")
    return weak


def _cert_margin(meta: dict):
    """-> ticket rate minus Gaussian rate on the certification layouts, or None.

    The one number that is comparable across bundles searched on different
    machines under different settings. A search rate is not: it is a maximum
    over however many candidates that run drew, on whichever layouts its
    geometry happened to assign. Certification runs the ticket and the
    per-chunk draw over the SAME unselected layouts, so its margin means the
    same thing wherever it was measured.
    """
    c = meta.get("certification")
    if not c or c.get("ticket_rate") is None or c.get("gaussian_rate") is None:
        return None
    return float(c["ticket_rate"]) - float(c["gaussian_rate"])


def _describe(meta: dict) -> str:
    b = meta.get("beats_baseline")
    tag = {True: "BEATS", False: "weak", None: "unresolved"}[
        b if b in (True, False) else None]
    m = _cert_margin(meta)
    bits = [f"search {meta.get('search_success', '?')}", tag]
    if m is not None:
        c = meta["certification"]
        bits.append(f"certified {c['ticket']} vs {c['gaussian']} ({m:+.0%})")
    else:
        bits.append("not certified")
    if meta.get("cycle", 1) != 1:
        bits.append(f"cycle m={meta['cycle']}")
    return ", ".join(bits)


def _copy_pool(key, src_dir, out) -> str | None:
    """Bring the _done_*.npz for one key along with its ticket.

    The bundle carries one vector per task; the pool carries every candidate
    the search drew, and two tools need it -- try_runners mines the runner-ups
    for a clean sweep of the reported layouts, and plot_tickets reads it for
    three of its four panels, looking for it BESIDE the bundle. A merged
    directory without the pools is a bundle that cannot be examined or mined,
    and after merging four machines nobody remembers which directory held
    which task.
    """
    import shutil
    suite, tid = key.rsplit(".", 1)
    name = f"_done_{suite}_t{int(tid)}.npz"
    src = Path(src_dir)
    src = src.parent if src.is_file() else src
    f = src / name
    if not f.exists():
        return None
    dst = Path(out) / name
    if dst.exists() and dst.stat().st_size == f.stat().st_size:
        return name
    shutil.copy2(f, dst)
    return name


def merge(out_dir, *in_dirs, prefer_certified: bool = False,
          pools: bool = True):
    """Combine bundles from separate searches into one.

    Duplicate keys are COLLECTED, not raised on the first one. Splitting one
    suite across four machines makes collisions the normal case, and failing on
    the first means four round trips to see all of them.
    """
    tensors, info, src, missing, dups = {}, {}, {}, [], {}
    for d in in_dirs:
        try:
            t, m = load_bundle(d)
        except FileNotFoundError:
            # Searches finish at different times and merging is how a partial
            # set gets evaluated, so an empty directory is an ordinary state.
            missing.append(d)
            continue
        for k in t:
            if k in tensors:
                dups.setdefault(k, [(src[k], info.get(k, {}))]).append(
                    (d, m.get(k, {})))
                continue
            tensors[k], info[k], src[k] = t[k], m.get(k, {}), d

    if dups:
        unresolved = {}
        for k, sides in dups.items():
            margins = [_cert_margin(mm) for _, mm in sides]
            if prefer_certified and all(x is not None for x in margins):
                best = max(range(len(sides)), key=lambda i: margins[i])
                d_, m_ = sides[best]
                t_, _ = load_bundle(d_)
                tensors[k], info[k], src[k] = t_[k], m_, d_
                print(f"  {k}: kept {d_} on certification margin "
                      f"{margins[best]:+.0%}")
                for i, (dd, mm) in enumerate(sides):
                    if i != best:
                        print(f"      dropped {dd} ({margins[i]:+.0%})")
            else:
                unresolved[k] = sides
        if unresolved:
            msg = [f"{len(unresolved)} key(s) defined by more than one input; "
                   f"picking by argument order would make the result depend on "
                   f"how you typed the command."]
            for k, sides in sorted(unresolved.items()):
                msg.append(f"  {k}")
                for dd, mm in sides:
                    msg.append(f"    {dd}\n      {_describe(mm)}")
            msg.append("Delete the ones you do not want, or pass "
                       "--prefer-certified to resolve by certification margin "
                       "-- the only number comparable across separately "
                       "configured searches. Keys where a side is uncertified "
                       "cannot be resolved that way.")
            raise ValueError("\n".join(msg))

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # A HAND DECISION OUTLIVES THE MERGE. `disable` records a judgement made
    # from an eval -- libero_10.1 searched 14/15 and then cost 20 points on
    # the reported layouts -- and the source bundle it came from knows nothing
    # about that. Merging the sources back over the destination silently put
    # the ticket back in service and took two points off the suite.
    try:
        _prev_t, _prev_m = load_bundle(out)
    except FileNotFoundError:
        _prev_t, _prev_m = {}, {}
    kept_decisions = []
    for k, pm in _prev_m.items():
        if k in info and pm.get("disabled_note") and not info[k].get("disabled_note"):
            info[k] = dict(info[k], beats_baseline=False,
                           disabled_note=pm["disabled_note"])
            kept_decisions.append(k)
    if _prev_t:
        print(f"note: {out} already held {len(_prev_t)} ticket(s); they are "
              f"being replaced by the inputs.")
    save_file(tensors, str(out / BUNDLE))
    (out / META).write_text(json.dumps(info, indent=1, sort_keys=True))
    for k in kept_decisions:
        print(f"  kept the hand-disable on {k} -- the inputs do not know it "
              f"lost on the reported layouts")
    copied, nopool = [], []
    if pools:
        for k in sorted(tensors):
            (copied if _copy_pool(k, src[k], out) else nopool).append(k)
    print(f"{len(tensors)} tickets -> {out / BUNDLE}")
    for k in sorted(tensors):
        b = info.get(k, {}).get("beats_baseline")
        tag = {True: "", False: "   (weak -- eval uses Gaussian)",
               None: "   (unresolved)"}[b if b in (True, False) else None]
        shp = tuple(tensors[k].shape)
        cyc = f"   cycle m={shp[0]}" if len(shp) == 3 else ""
        print(f"  {k:<24} from {src[k]}{cyc}{tag}")
    for d in missing:
        print(f"  (no bundle yet in {d})")
    if pools:
        mb = sum((out / f"_done_{k.rsplit('.', 1)[0]}_t{int(k.rsplit('.', 1)[1])}.npz")
                 .stat().st_size for k in copied) / 1e6
        print(f"copied {len(copied)} candidate pool(s), {mb:.1f} MB, so "
              f"try_runners and plot_tickets work against this directory")
        if nopool:
            print(f"  no pool found for {', '.join(nopool)} -- those tickets "
                  f"cannot be mined for runner-ups from here")
    return out / BUNDLE


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "revert":
        # Rebuild the held-out bundle. try_runners selects candidates by
        # running them on the layouts the benchmark reports, which is exact
        # but in-sample, and it saves what it replaced to replaced.json. A
        # paper wants both numbers -- spatial reads 97.5 with three mined
        # tickets in and about 95.0 without -- and without this the held-out
        # one is only recoverable by re-running the searches.
        import numpy as _np
        d = Path(sys.argv[2])
        keys = sys.argv[3:]
        log = d / "replaced.json"
        if not log.exists():
            print(f"no replaced.json in {d}; nothing was ever replaced here",
                  file=sys.stderr)
            raise SystemExit(1)
        hist = json.loads(log.read_text())
        t, m = load_bundle(d)
        todo = keys or sorted(hist)
        for k in todo:
            if k not in hist or not hist[k]:
                print(f"  {k}: no replacement on record")
                continue
            last = hist[k][-1]
            f = d / last["vector"]
            if not f.exists():
                print(f"  {k}: {last['vector']} is gone", file=sys.stderr)
                continue
            suite_, tid_ = k.rsplit(".", 1)
            save_ticket(d, suite_, int(tid_), _np.load(f), last["meta"])
            print(f"  {k}: restored the ticket replaced on {last['when']} "
                  f"({_describe(last['meta'])})")
        print("\nRe-run the eval; this bundle is the held-out one again.")
        raise SystemExit(0)
    if len(sys.argv) >= 5 and sys.argv[1] == "put":
        # Put a vector or an m-tuple into a bundle under a key. The search
        # writes its own winners; this is for one measured by hand -- a random
        # m=4 cycle that matched the per-chunk draw on goal T9 where the
        # searched single ticket scored 15 points below it. Records that it
        # was not searched, so `report` cannot present it as one.
        import numpy as _np
        d, k, f = sys.argv[2], sys.argv[3], sys.argv[4]
        suite_, tid_ = k.rsplit(".", 1)
        v = _np.load(f)
        meta = {"task": None, "beats_baseline": True,
                "search_success": "not searched",
                "baseline_search": "n/a",
                "source": f, "cycle": int(v.shape[0]) if v.ndim == 3 else 1,
                "hand_placed": True,
                "note": " ".join(sys.argv[5:]) or "placed by hand"}
        print(f"  {k}: shape {tuple(v.shape)}"
              + (f", a {v.shape[0]}-vector cycle" if v.ndim == 3 else ""))
        print(f"  -> {save_ticket(d, suite_, int(tid_), v, meta)}")
        raise SystemExit(0)
    if len(sys.argv) >= 4 and sys.argv[1] in ("disable", "enable"):
        # NOT a delete. A ticket that loses to Gaussian on the reported
        # layouts is still the output of a search that has been paid for, and
        # the call can be wrong: goal T9's ticket reads 11/20 against a
        # Gaussian draw of 14/20, which is p=0.13, not proof. Flipping
        # beats_baseline makes eval fall back to Gaussian while the vector,
        # the search record and the decision stay on disk and reversible.
        d, keys = sys.argv[2], sys.argv[3:]
        t, m = load_bundle(d)
        want = sys.argv[1] == "enable"
        for k in keys:
            if k not in t:
                print(f"  {k}: not in this bundle", file=sys.stderr)
                continue
            was = m.setdefault(k, {}).get("beats_baseline")
            m[k]["beats_baseline"] = True if want else False
            m[k]["disabled_note" if not want else "enabled_note"] = (
                "beats_baseline set by hand; see the eval that motivated it")
            print(f"  {k}: beats_baseline {was} -> {m[k]['beats_baseline']}"
                  + ("   eval will use Gaussian" if not want
                     else "   eval will use the ticket"))
        (Path(d) / META).write_text(json.dumps(m, indent=1, sort_keys=True))
        raise SystemExit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "runners":
        # THE BUNDLE KEEPS ONE TICKET; THE SEARCH KEPT ALL OF THEM. Sequential
        # halving usually leaves several candidates tied at the top tier's
        # score, and only the first was banked. When the banked one turns out
        # not to solve every canonical layout, a tied runner-up may -- and
        # testing one is 35 episodes (the layouts it has never run) against
        # hours for a fresh search.
        import numpy as _np
        z = _np.load(sys.argv[2], allow_pickle=True)
        w, r, cands = z["wins"], z["runs"], z["cands"]
        rate = _np.where(r > 0, w / _np.maximum(r, 1), -1.0)
        # DEPTH FIRST, then rate. Ranking by rate alone put candidates that
        # were eliminated in tier 1 above the deep survivors: object T5 listed
        # six tickets at 2/3 -- three layouts each, barely tested -- above the
        # 8/15 that the search actually banked, and object T3 listed a 7/9
        # above its 9/15. Runs differ BECAUSE the search stopped the weak ones
        # early, so a high rate over few layouts is the absence of evidence.
        order = sorted(range(len(rate)), key=lambda i: (-r[i], -rate[i]))
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        top = [i for i in order if r[i] > 0][:k]
        deep = max(r)
        # The banked one is named in the bundle metadata, not inferred from the
        # ranking -- which is what produced the wrong "<- banked" marks.
        banked = None
        d = Path(sys.argv[2])
        m = d.parent / META
        if m.exists():
            stem = d.stem.replace("_done_", "")
            suite_, tid_ = stem.rsplit("_t", 1)
            info = json.loads(m.read_text()).get(key(suite_, int(tid_)), {})
            banked = info.get("ticket_index")
        g = json.loads(str(z["geom"])) if "geom" in z.files else None
        print(f"{str(z['desc'])}")
        if g:
            print(f"  searched with {g.get('tickets')} tickets, "
                  f"{g.get('envs_per_tier')} layouts per tier, from id "
                  f"{g.get('init_state_offset')}")
        print(f"  deepest candidates ran {int(deep)} layouts\n")
        print(f"{'rank':>4}  {'ticket':>6}  {'score':>7}  rate")
        for n, i in enumerate(top):
            mark = "   <- banked" if banked is not None and i == banked else ""
            if r[i] < deep:
                mark += f"   (only {int(r[i])} layouts -- eliminated early)"
            print(f"{n:>4}  {i:>6}  {int(w[i]):>3}/{int(r[i]):<3}  "
                  f"{rate[i]:.0%}{mark}")
        at_depth = [i for i in range(len(r)) if r[i] == deep]
        best_at_depth = max(rate[i] for i in at_depth)
        tied = [i for i in at_depth if rate[i] >= best_at_depth - 1e-9]
        print(f"\n{len(at_depth)} candidate(s) reached {int(deep)} layouts; "
              f"{len(tied)} of them tied at the top score "
              f"{int(w[tied[0]])}/{int(deep)}.")
        if best_at_depth < 1.0:
            print("  The best full-depth candidate is not perfect even on the "
                  "layouts it was searched on, so no runner-up here is a "
                  "likely 20/20. Re-searching this task is the honest option.")
        if len(sys.argv) > 4 and sys.argv[4] == "--export":
            out = Path(sys.argv[2]).parent
            for n, i in enumerate(tied):
                f = out / f"{Path(sys.argv[2]).stem.replace('_done_', 'alt_')}_c{i}.npy"
                _np.save(f, cands[i])
                print(f"  wrote {f}")
            print("\nTest one with eval_libero.py --noise_ticket <file> "
                  "--task_ids <id> --episodes 20 --init_state_offset 0,\n"
                  "then again at --init_state_offset 35 --episodes 15. A "
                  "candidate that takes both is 50/50.")
        raise SystemExit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "report":
        raise SystemExit(0 if report(sys.argv[2]) == 0 else 0)
    if sys.argv[1:2] == ["merge"] and len(sys.argv) >= 4:
        flags = {"--prefer-certified", "--no-pools"}
        args = [x for x in sys.argv[2:] if x not in flags]
        merge(args[0], *args[1:],
              prefer_certified="--prefer-certified" in sys.argv,
              pools="--no-pools" not in sys.argv)
        raise SystemExit(0)
    print("usage: python ticket_bundle.py report <dir>\n"
          "       python ticket_bundle.py disable|enable <dir> <suite.task> ...\n"
          "       python ticket_bundle.py put <dir> <suite.task> <vec.npy> [note...]\n"
          "       python ticket_bundle.py revert <dir> [suite.task ...]\n"
          "       python ticket_bundle.py runners <_done_*.npz> [k] [--export]\n"
          "       python ticket_bundle.py merge <out_dir> <in_dir> ... "
          "[--prefer-certified] [--no-pools]",
          file=sys.stderr)
    raise SystemExit(2)
