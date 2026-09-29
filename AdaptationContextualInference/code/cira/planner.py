"""Cross-entropy-method planner minimizing the belief-weighted cost sum_c pi(c) J_c(a) (paper Sec. 3.5)."""

from __future__ import annotations

import numpy as np
import torch

from cira.config import EnvConfig, PlannerConfig
from cira.frozen import FrozenModel, features


class CEMPlanner:
    def __init__(self, frozen: FrozenModel, env_cfg: EnvConfig, cfg: PlannerConfig):
        self.f, self.env_cfg, self.cfg = frozen, env_cfg, cfg
        self.lo = np.array(env_cfg.action_low, float)
        self.hi = np.array(env_cfg.action_high, float)
        self._a_gain = torch.as_tensor(frozen.a_gain, dtype=torch.float32)
        self._d_sd = torch.as_tensor(frozen.d_sd, dtype=torch.float32)
        self._target = torch.as_tensor(env_cfg.target, dtype=torch.float32)

    @torch.no_grad()
    def rollout(self, z0: np.ndarray, actions: np.ndarray, Ws: np.ndarray) -> torch.Tensor:
        """Predicted puck positions (H, N, T+1, 2) for N actions under H residual hypotheses."""
        H, N = len(Ws), len(actions)
        z = torch.as_tensor(z0, dtype=torch.float32).expand(H, N, 4).clone()
        a0 = torch.as_tensor(actions, dtype=torch.float32).expand(H, N, 2)
        zero = torch.zeros_like(a0)
        W = torch.as_tensor(Ws, dtype=torch.float32)  # (H, k, p)
        moving = torch.ones(H, N, 1, dtype=torch.bool)
        pos = [z[..., :2]]
        for t in range(self.cfg.horizon):
            a = a0 if t == 0 else zero
            dfn0 = self.f.delta_norm_t(z, zero)
            dfn = self.f.delta_norm_t(z, a) if t == 0 else dfn0
            corr = torch.einsum("hnp,hkp->hnk", features(a * self._a_gain, dfn, dfn0), W)
            z_new = z + (dfn + corr) * self._d_sd
            if t > 0:
                # A puck at rest stays at rest: stop once the predicted speed is small or reverses.
                v_old, v_new = z[..., 2:], z_new[..., 2:]
                stopped = (v_new.norm(dim=-1, keepdim=True) < self.env_cfg.stop_speed) | (
                    (v_old * v_new).sum(-1, keepdim=True) < 0
                )
                moving = moving & ~stopped
                z_new = torch.where(moving, z_new, torch.cat([z[..., :2], torch.zeros_like(v_old)], -1))
            z = z_new
            pos.append(z[..., :2])
            if t > 0 and not moving.any():
                pos.extend([z[..., :2]] * (self.cfg.horizon - 1 - t))
                break
        return torch.stack(pos, 2)

    def cost(self, pos: torch.Tensor) -> torch.Tensor:
        """Distance of the resting position to the target; leaving the table costs a constant plus overshoot."""
        c = self.env_cfg
        x, y = pos[..., 0], pos[..., 1]
        over = (x - c.edge_x).clamp(min=0) + (-c.half_l - x).clamp(min=0) + (y.abs() - c.half_w).clamp(min=0)
        off = over.amax(-1)
        dist = (pos[..., -1, :] - self._target).norm(dim=-1)
        return torch.where(off > 0, self.cfg.off_table_cost + off, dist)

    def plan(self, z0: np.ndarray, hyps, rng: np.random.Generator):
        hyps = sorted(hyps, key=lambda h: -h[0])[: self.cfg.max_hypotheses]
        probs = torch.as_tensor([p for p, _ in hyps], dtype=torch.float32)
        probs = probs / probs.sum()
        Ws = np.stack([W for _, W in hyps])
        cfg = self.cfg
        mean = (self.lo + self.hi) / 2
        std = cfg.init_std_frac * (self.hi - self.lo)
        best_a, best_c = mean, np.inf
        for _ in range(cfg.iters):
            A = np.clip(mean + std * rng.standard_normal((cfg.samples, 2)), self.lo, self.hi)
            J = (probs[:, None] * self.cost(self.rollout(z0, A, Ws))).sum(0).numpy()
            order = np.argsort(J)
            elite = A[order[: cfg.elites]]
            mean, std = elite.mean(0), elite.std(0) + 1e-3 * (self.hi - self.lo)
            if J[order[0]] < best_c:
                best_a, best_c = A[order[0]].copy(), float(J[order[0]])
        pred = self.rollout(z0, best_a[None], Ws)[:, 0].numpy()  # (H, T+1, 2) per hypothesis
        return best_a, pred, best_c, probs.numpy()
