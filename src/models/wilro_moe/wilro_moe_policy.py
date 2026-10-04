import math
from collections import deque

import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from .wilro_moe_model import WilroMoETransformer
from .wilro_moe_config import WilroMoEConfig


class WilroMoEPolicy(PreTrainedPolicy):
    """LeRobot wrapper for WILRO-MoE (SmolVLM2 KV-cache -> expert decoders)."""

    config_class = WilroMoEConfig
    name = "wilro_moe"

    def __init__(self, config: WilroMoEConfig):
        super().__init__(config)
        self.config = config
        self.model = WilroMoETransformer(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        # Chunked execution: commit n_action_steps, re-sample only when drained.
        # Each re-sample draws fresh flow noise, which at small n_action_steps is
        # where most of this family's success rate comes from.
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        # Temporal-ensembling state. Buffer holds (start_step, (B, H, D)) for
        # every chunk still covering the current timestep.
        self._te_buf: deque = deque()
        self._te_t = 0
        # Where a --noise_cycle restarts. Without this the cycle phase would
        # depend on how many chunks the PREVIOUS episode used, so the same
        # layout would replay differently depending on what ran before it --
        # and a deterministic replay is the whole reason for a fixed cycle.
        self.model._noise_cycle_k = 0
        # Stall escape state, per env.
        self._stall_prev = None
        self._stall_max = None
        self._stall_count = None
        # Read by eval_libero to count chunks and to gate the per-chunk
        # motion accumulator. Its old test -- `not policy._action_queue` -- is
        # always True under temporal ensembling, which would inflate the chunk
        # count by n_action_steps and change what the 2 mm still-threshold is
        # measured over.
        self._drew_chunk = False

    def forward(self, batch: dict) -> tuple:
        loss = self.model.compute_loss(batch)
        w = float(getattr(self.config, "router_balance_weight", 0.0) or 0.0)
        if w > 0.0:
            bal = self.model.router_balance_loss()
            if bal is not None:
                loss = loss + w * bal
        return loss, {}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        self.model.eval()
        return self.model.sample_actions(batch)

    # ------------------------------------------------------------------
    # Stall escape
    # ------------------------------------------------------------------
    def _stall_scale(self, batch: dict, B: int, device, dtype):
        """-> (B,) noise scale, or None when the feature is off.

        "Still" is judged RELATIVE to the largest state step this episode has
        produced, so it needs no unit conversion and no dataset stats: the
        observation reaching select_action is already normalized.
        """
        hi = float(getattr(self.config, "stall_noise_scale", 0.0) or 0.0)
        if hi <= 0.0:
            return None
        st = batch.get("observation.state")
        if st is None:
            return None
        st = (st[:, -1] if st.dim() == 3 else st).detach().float()

        base = float(getattr(self.config, "sample_noise_scale", 1.0) or 1.0)
        scale = torch.full((B,), base, device=device, dtype=dtype)
        if self._stall_prev is None or self._stall_prev.shape != st.shape:
            self._stall_prev = st
            self._stall_max = torch.zeros(B, device=st.device, dtype=st.dtype)
            self._stall_count = torch.zeros(B, device=st.device, dtype=torch.long)
            return scale

        step = (st - self._stall_prev).norm(dim=-1)
        self._stall_prev = st
        self._stall_max = torch.maximum(self._stall_max, step)
        still = step < float(self.config.stall_rel_threshold) * self._stall_max
        self._stall_count = torch.where(
            still, self._stall_count + 1, torch.zeros_like(self._stall_count))
        fire = self._stall_count >= int(self.config.stall_patience)
        return torch.where(fire.to(device), torch.full_like(scale, hi), scale)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        self.model.eval()
        coeff = float(getattr(self.config, "temporal_ensemble_coeff", 0.0) or 0.0)
        if coeff <= 0.0:
            # Unchanged path, bit-identical to before temporal ensembling existed.
            self._drew_chunk = not self._action_queue
            if len(self._action_queue) == 0:
                actions = self.model.sample_actions(batch)[:, : self.config.n_action_steps]
                self._action_queue.extend(actions.transpose(0, 1))
            return self._action_queue.popleft()

        # Temporal ensembling. The draw cadence is UNCHANGED -- a new chunk every
        # n_action_steps -- so this costs no extra forward passes. What changes is
        # that the older chunks' predictions for the current timestep are averaged
        # in instead of discarded. With horizon 64 and n_action_steps 2 that is up
        # to 32 independent noise draws per emitted action.
        H = int(self.config.horizon)
        n_exec = max(1, int(self.config.n_action_steps))
        self._drew_chunk = (self._te_t % n_exec == 0)
        if self._drew_chunk:
            B = batch["observation.state"].shape[0]
            dev = batch["observation.state"].device
            scale = self._stall_scale(batch, B, dev, torch.float32)
            self.model._noise_scale_override = scale
            try:
                chunk = self.model.sample_actions(batch, full=True)
            finally:
                self.model._noise_scale_override = None
            if chunk.shape[1] < H:
                raise RuntimeError(
                    f"temporal ensembling needs the full horizon: sample_actions "
                    f"returned {chunk.shape[1]} steps, config.horizon is {H}. "
                    f"Pass full=True.")
            if self._te_buf and self._te_buf[-1][1].shape[0] != chunk.shape[0]:
                # Batch width changed under us; the aligned buffer is meaningless.
                self._te_buf.clear()
            self._te_buf.append((self._te_t, chunk))

        # Drop chunks whose horizon no longer reaches the current step.
        while self._te_buf and self._te_t - self._te_buf[0][0] >= H:
            self._te_buf.popleft()

        num = None
        den = 0.0
        for start, chunk in self._te_buf:
            age = self._te_t - start
            w = math.exp(-coeff * age)
            term = chunk[:, age] * w
            num = term if num is None else num + term
            den += w
        self._te_t += 1
        return num / den
