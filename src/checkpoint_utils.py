"""Resolve a checkpoint argument that may be a local path or a Hub repo id.

Lives here rather than in train_wiltechs_x.py because eval_libero.py needs
it, and importing the trainer to get it dragged in the whole WiltechsVLA ->
Qwen3-VL chain. That made the harness unusable in exactly the places it is most
needed: a worktree pinned to a commit predating those models, and a separate
environment built only for a candidate teacher policy --

    from transformers import Qwen3VLForConditionalGeneration
    ModuleNotFoundError: No module named 'transformers'

-- for a function whose only dependencies are pathlib and huggingface_hub.
Same reason src/libero_paraphrase.py is not inside models/wiltechs_x/.
"""
from __future__ import annotations

from pathlib import Path


def resolve_checkpoint(path_or_repo: str, *, for_resume: bool = True) -> Path:
    """Accept a local directory OR a Hugging Face repo id.

    `--resume_from_checkpoint ISdept/wiltech-x-6k` used to fail with a bare
    FileNotFoundError on `<repo>/training_state.pth`, which reads like a
    corrupt checkpoint rather than "that is a Hub id and this function only
    knew about disk".

    `for_resume` picks what to download. Resuming reads training_state.pth,
    which already carries the model, so model.safetensors would be ~9.5 GiB of
    waste. Loading a policy for eval or a probe needs exactly the opposite.
    """
    p = Path(path_or_repo)
    if p.exists():
        return p
    if "/" not in path_or_repo or path_or_repo.startswith((".", "/", "~")):
        raise SystemExit(f"no such checkpoint directory: {path_or_repo}")
    from huggingface_hub import list_repo_files, snapshot_download

    print(f"'{path_or_repo}' is not a local path; fetching from the Hub...")
    files = set(list_repo_files(path_or_repo))
    if for_resume and "training_state.pth" in files:
        # NOT lerobot's allow_patterns (*.safetensors/*.json): that filter
        # drops training_state.pth and would silently downgrade a full resume
        # to a weights-only one. And since the .pth already carries the model,
        # pulling model.safetensors too is ~9.5 GiB of redundant download.
        patterns = ["training_state.pth", "*.json"]
        print("  repo has training_state.pth -> full resume "
              "(skipping the redundant model.safetensors)")
    elif for_resume:
        patterns = ["*.safetensors", "*.json"]
        print("  repo has no training_state.pth -> weights-only resume")
    else:
        patterns = ["*.safetensors", "*.json"]
    local = Path(snapshot_download(repo_id=path_or_repo, allow_patterns=patterns))
    print(f"  -> {local}")
    return local
