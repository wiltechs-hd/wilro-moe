"""Record the library versions that can silently change what a number means.

WHY THIS EXISTS
---------------
Every comparison in notes/libero_benchmark_tracker.md is PAIRED: same layouts,
same seed, same flow-noise stream, so two checkpoints differ only where the
policy does. `eval_commit` pins the repo. Nothing pinned the environment, and
the environment is just as capable of moving a number:

  * 2026-09-17 the training box went L4 -> A100. Different kernels mean a
    different floating-point reduction order, which over 10 ODE steps and 280
    closed-loop chunks is a perturbation of a chaotic system. Eval was kept on
    the L4 precisely so this could not reach the tracker.
  * 2026-09-18 a Colab session came back on Python 3.13 instead of 3.10 and
    torchcodec could no longer decode the dataset's videos. That one at least
    CRASHED. A decoder that merely rounds differently would not have.

A crash is the lucky case. The dangerous one is an environment that still runs
and quietly shifts the numbers, because the paired tests have no way to see it.
So: stamp the fingerprint into every result file, and print it at startup.

`digest` is the first 8 hex of a hash over the whole dict -- two runs with the
same digest were measured under the same environment, and that is checkable at
a glance instead of by diffing.
"""
import hashlib
import json
import platform

_PKGS = ("torch", "lerobot", "robosuite", "mujoco", "libero", "torchcodec",
         "av", "numpy", "transformers", "gymnasium")


def _ver(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        pass
    try:                                     # packages that ship no metadata
        import importlib
        return getattr(importlib.import_module(name), "__version__", None)
    except Exception:
        return None


def fingerprint() -> dict:
    """-> {python, gpu, cuda, <package>: version, digest}. Never raises."""
    fp: dict = {"python": platform.python_version()}
    for p in _PKGS:
        v = _ver(p)
        if v is not None:
            fp[p] = v
    try:
        import torch
        fp["cuda"] = torch.version.cuda
        # The GPU belongs here, not in a comment: it decides the kernels, and
        # kernels decide the reduction order.
        fp["gpu"] = (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else "cpu")
    except Exception:
        pass
    fp["digest"] = hashlib.sha256(
        json.dumps(fp, sort_keys=True).encode()).hexdigest()[:8]
    return fp


def fingerprint_line(fp: dict | None = None) -> str:
    fp = fp or fingerprint()
    keys = ("gpu", "python", "torch", "lerobot", "robosuite", "mujoco",
            "torchcodec", "av")
    body = "  ".join(f"{k}={fp[k]}" for k in keys if k in fp)
    return f"[env {fp['digest']}] {body}"


if __name__ == "__main__":
    print(fingerprint_line())
    print(json.dumps(fingerprint(), indent=2))
