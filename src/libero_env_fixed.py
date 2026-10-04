"""Correct LIBERO's init-state ordering in lerobot's LiberoEnv.

lerobot's LiberoEnv.reset() does:

    set_init_state(_init_states[_init_state_id])   # write MuJoCo state
    _env.reset()                                   # robosuite _reset_internal()

robosuite's reset re-samples the placement initializer, discarding the state
that was just written. LIBERO's own eval loop is the other way round:

    env.reset()
    env.set_init_state(init_states[i])
    for _ in range(num_steps_wait): env.step(dummy)

Measured on libero_spatial task 0 (src/libero_init_state_check.py), per-object
xyz spread over 12 layouts, metres:

    object                   canonical   lerobot today   fixed order   fixed+same id
    akita_black_bowl_1          0.0089          0.0876        0.0143          0.0000
    plate_1                     0.0080          0.0503        0.0076          0.0000
    ramekin_1                   0.0094          0.0284        0.0092          0.0000

The fixed order is exactly deterministic and reproduces the canonical layouts.
Today's order serves layouts 3-10x more spread out than the ones the LIBERO
demos were recorded on -- so training sees a +-1.5cm jitter while eval draws
from a distribution ~10x wider, and no seed or init_state_id pins it down.

Two consequences beyond measurement:

  * GRPO groups break. train_wiltechs_vla_rl.TaskEnvGroup.reset_group() sets one
    _init_state_id for all G members but a different seed each, on the premise
    that "a GRPO group shares the same initial state". Under today's ordering the
    seed drives the placement sampler, so the G members get DIFFERENT layouts and
    the group-mean baseline absorbs layout difficulty instead of action quality.
  * Numbers are not comparable to published LIBERO results, which are reported
    on the canonical 50 states.

Usage -- patch once, before any env is built, and everything downstream (your
eval, lerobot's make_libero_envs, the RL scripts, the probes) is covered:

    from libero_env_fixed import patch_lerobot_libero
    patch_lerobot_libero()

To A/B the same checkpoint without touching eval code, gate it on an env var:

    LIBERO_FIXED_INIT=0 python your_eval.py    # today's behaviour
    LIBERO_FIXED_INIT=1 python your_eval.py    # canonical layouts

patch_lerobot_libero() reads LIBERO_FIXED_INIT when called with no argument and
defaults to on. Or subclass explicitly with FixedOrderLiberoEnv.
"""

import os

_ORIGINAL_RESET = None
_INIT_FROM_SEED = False


def _fixed_reset(self, seed=None, **kwargs):
    """LIBERO's canonical order: reset, then write the init state, then settle."""
    import gymnasium as gym
    from lerobot.envs.libero import get_libero_dummy_action

    gym.Env.reset(self, seed=seed)
    self._env.seed(seed)
    if (_INIT_FROM_SEED and seed is not None and self.init_states
            and self._init_states is not None):
        # Opt-in, eval only. create_libero_envs pins _init_state_id to the
        # sub-env's episode_index for its whole life, so an eval with
        # batch_size=6 would only ever visit states 0..5 no matter how many
        # episodes it runs. lerobot_eval hands each episode a unique seed
        # (start_seed + batch_ix*num_envs + i), which makes a good index.
        # NOT for RL: TaskEnvGroup gives each group member a different seed on
        # purpose, so deriving the state from it would put every member on a
        # different layout -- the very thing the ordering fix repairs.
        self._init_state_id = int(seed) % len(self._init_states)
    raw_obs = self._env.reset()
    if self.init_states and self._init_states is not None:
        # Overwrites whatever the placement sampler just produced. Must come
        # after _env.reset(), which is the entire point of this module.
        self._env.set_init_state(self._init_states[self._init_state_id])
    # Objects can be fractionally interpenetrating right after a state write, so
    # settle exactly as the stock reset does -- same num_steps_wait, same dummy
    # action -- to keep the two orderings comparable.
    for _ in range(self.num_steps_wait):
        raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
    return self._format_raw_obs(raw_obs), {"is_success": False}


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False")


def patch_lerobot_libero(enable: bool | None = None,
                         init_from_seed: bool | None = None) -> bool:
    """Swap LiberoEnv.reset in place. Returns whether the fix is now active.

    Idempotent, and reversible by calling with enable=False -- so an A/B can run
    in one process. Patching the class rather than exposing a subclass means
    code that builds LiberoEnv itself (lerobot.envs.libero.make_libero_envs, and
    whatever your eval notebook does) picks it up without edits.

    init_from_seed derives _init_state_id from the per-episode seed so an eval
    covers more than batch_size layouts. Off by default; see _fixed_reset for
    why it must stay off for RL. Reads LIBERO_INIT_FROM_SEED (default 0).
    """
    global _ORIGINAL_RESET, _INIT_FROM_SEED
    from lerobot.envs.libero import LiberoEnv

    if enable is None:
        enable = _flag("LIBERO_FIXED_INIT", "1")
    _INIT_FROM_SEED = (_flag("LIBERO_INIT_FROM_SEED", "0")
                       if init_from_seed is None else init_from_seed)

    if _ORIGINAL_RESET is None:
        _ORIGINAL_RESET = LiberoEnv.reset

    if enable:
        LiberoEnv.reset = _fixed_reset
        print("[libero] init-state ordering: FIXED (reset -> set_init_state). "
              "Layouts are the canonical 50 and _init_state_id selects them.")
        if _INIT_FROM_SEED:
            print("[libero] _init_state_id derived from the per-episode seed "
                  "(eval only -- do NOT use for RL group rollouts).")
    else:
        LiberoEnv.reset = _ORIGINAL_RESET
        print("[libero] init-state ordering: STOCK (set_init_state -> reset). "
              "Layouts come from the placement sampler; _init_state_id is not "
              "honoured.")
    return enable


def make_fixed_env_class():
    """Explicit subclass, for call sites that would rather not monkeypatch."""
    from lerobot.envs.libero import LiberoEnv

    class FixedOrderLiberoEnv(LiberoEnv):
        reset = _fixed_reset

    return FixedOrderLiberoEnv



def get_sim_env(libero_env):
    """Walk LiberoEnv -> OffScreenRenderEnv -> robosuite problem env (the object
    that has both .sim and _check_success). Returns None if not found."""
    cur = libero_env
    seen = {id(cur)}
    for _ in range(12):
        if getattr(cur, "sim", None) is not None and hasattr(cur, "_check_success"):
            return cur
        nxt = None
        for attr in ("env", "_env", "unwrapped"):
            cand = getattr(cur, attr, None)
            if cand is not None and id(cand) not in seen:
                nxt = cand
                break
        if nxt is None:
            break
        seen.add(id(nxt))
        cur = nxt
    if getattr(cur, "sim", None) is not None and hasattr(cur, "_check_success"):
        return cur
    return None


def verify(suite_name: str = "libero_spatial", task_id: int = 0, n: int = 4) -> bool:
    """Assert the patch actually pins the layout. Resets twice per init state and
    checks the object positions match to <1mm, then checks two different states
    differ. Cheap enough to call at the top of an eval run."""
    import numpy as np
    from lerobot.envs.libero import LiberoEnv, _get_suite

    env = LiberoEnv(task_suite=_get_suite(suite_name), task_id=task_id,
                    task_suite_name=suite_name, obs_type="pixels_agent_pos",
                    init_states=True, episode_index=0)
    try:
        se = get_sim_env(env)
        if se is None:
            print("[libero] verify: could not reach the sim; skipping.")
            return False

        def pos():
            bid = getattr(se, "obj_body_id", None) or {}
            return {k: np.array(se.sim.data.body_xpos[v]) for k, v in bid.items()}

        seen = {}
        repeatable = True
        for i in range(n):
            for rep in range(2):
                env._init_state_id = i
                env.reset(seed=1000 * rep + i)   # different seeds on purpose
                p = pos()
                if rep == 0:
                    seen[i] = p
                else:
                    d = max(np.linalg.norm(p[k] - seen[i][k]) for k in p)
                    if d > 1e-3:
                        print(f"[libero] verify: state {i} not repeatable across "
                              f"seeds ({d:.4f} m) -- the fix is NOT active.")
                        repeatable = False
        spread = max(np.linalg.norm(seen[0][k] - seen[i][k])
                     for i in range(1, n) for k in seen[0]) if n > 1 else 0.0
        print(f"[libero] verify: repeatable={repeatable}, "
              f"max layout difference between states = {spread:.4f} m")
        return repeatable
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--stock", action="store_true",
                    help="verify WITHOUT the fix, to see it fail")
    a = ap.parse_args()
    patch_lerobot_libero(enable=not a.stock)
    verify(a.suite, a.task_id, a.n)
