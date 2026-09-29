"""Frozen dynamics model f(z, a) trained on the training condition only, plus its noise model sigma(z, a).

Stage 1a works on the true low-dimensional state z = [x, y, vx, vy], so the residual subspace U is
the identity (k = 4). All residual quantities are expressed in normalized units: deltas are divided
by the per-coordinate std of training deltas.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cira.config import EnvConfig, FrozenConfig
from cira.env import PuckEnv


def mlp(n_in: int, n_out: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_out))


def collect_training_data(env_cfg: EnvConfig, n_episodes: int, seed: int = 0):
    """Transitions under the training condition: strikes from rest plus randomly moving pucks."""
    rng = np.random.default_rng(seed)
    env = PuckEnv(env_cfg)
    lo, hi = np.array(env_cfg.action_low), np.array(env_cfg.action_high)
    Z, A, Zn = [], [], []
    for _ in range(n_episodes):
        env.set_context(*env_cfg.train_context, rng)
        if rng.random() < 0.5:
            start = (rng.uniform(*env_cfg.start_x), rng.uniform(*env_cfg.start_y))
            env.reset(start)
            impulse = rng.uniform(lo, hi)
        else:
            start = (rng.uniform(-env_cfg.half_l + 0.05, env_cfg.edge_x - 0.05), rng.uniform(-0.3, 0.3))
            speed, ang = rng.uniform(0, 6.0), rng.uniform(0, 2 * np.pi)
            env.reset(start, (speed * np.cos(ang), speed * np.sin(ang)))
            impulse = rng.uniform(lo, hi) * rng.choice([-1.0, 1.0], size=2) if rng.random() < 0.5 else np.zeros(2)
        r = env.strike(impulse, rng, rest_steps=8)
        if len(r.actions):
            Z.append(r.states[:-1])
            A.append(r.actions)
            Zn.append(r.states[1:])
    return np.concatenate(Z), np.concatenate(A), np.concatenate(Zn)


class FrozenModel:
    def __init__(self, env_cfg: EnvConfig, hidden: int = 128, sigma_floor: float = 0.02):
        self.env_cfg = env_cfg
        self.mean_net = mlp(6, 4, hidden)
        self.noise_net = mlp(6, 4, hidden)
        self.sigma_floor = sigma_floor
        self.z_mu = np.zeros(4)
        self.z_sd = np.ones(4)
        self.a_scale = np.maximum(np.abs(env_cfg.action_low), np.abs(env_cfg.action_high)).astype(float)
        self.d_sd = np.ones(4)

    # ------------------------------------------------------------------ inference (torch)
    def _inp(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        zt = (z - torch.as_tensor(self.z_mu, dtype=z.dtype)) / torch.as_tensor(self.z_sd, dtype=z.dtype)
        return torch.cat([zt, a / torch.as_tensor(self.a_scale, dtype=a.dtype)], dim=-1)

    @torch.no_grad()
    def delta_norm_t(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.mean_net(self._inp(z, a))

    @torch.no_grad()
    def sigma_t(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.sigma_floor + nn.functional.softplus(self.noise_net(self._inp(z, a)))

    # ------------------------------------------------------------------ inference (numpy)
    def predict_norm(self, Z: np.ndarray, A: np.ndarray):
        z = torch.as_tensor(Z, dtype=torch.float32)
        a = torch.as_tensor(A, dtype=torch.float32)
        return self.delta_norm_t(z, a).numpy().astype(float), self.sigma_t(z, a).numpy().astype(float)

    @property
    def a_gain(self) -> np.ndarray:
        """Impulse -> normalized velocity change for the training mass (so action weights are O(1))."""
        m0 = self.env_cfg.loads[self.env_cfg.train_context[1]]
        return 1.0 / (m0 * self.d_sd[2:])

    def residuals(self, Z: np.ndarray, A: np.ndarray, Zn: np.ndarray):
        """Features phi, normalized residuals v and noise scales sigma for a batch of transitions."""
        dfn, sig = self.predict_norm(Z, A)
        dfn0, _ = self.predict_norm(Z, np.zeros_like(A))
        v = (Zn - Z) / self.d_sd - dfn
        c = self.env_cfg
        slow = np.minimum(np.linalg.norm(Z[:, 2:], axis=1), np.linalg.norm(Zn[:, 2:], axis=1)) < c.slow_speed
        slow &= ~np.any(A != 0, axis=1) | (np.linalg.norm(Zn[:, 2:], axis=1) < c.slow_speed)
        sig = np.where(slow[:, None], sig * c.slow_sigma_factor, sig)
        return features(A * self.a_gain, dfn, dfn0), v, sig

    # ------------------------------------------------------------------ training
    def fit(self, Z, A, Zn, cfg: FrozenConfig, log=print, device: str | None = None):
        """Fit on `device` (default: cuda if available); the model is returned on the CPU."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.z_mu, self.z_sd = Z.mean(0), Z.std(0) + 1e-6
        D = Zn - Z
        self.d_sd = D.std(0) + 1e-6
        X = self._inp(torch.as_tensor(Z, dtype=torch.float32), torch.as_tensor(A, dtype=torch.float32))
        Y = torch.as_tensor(D / self.d_sd, dtype=torch.float32)
        X, Y = X.to(device), Y.to(device)
        self.mean_net.to(device)
        self.noise_net.to(device)
        log(f"training on {device}")
        idx = rng.permutation(len(X))
        n_tr = int(0.8 * len(X))
        tr, va = idx[:n_tr], idx[n_tr:]

        def train(net, loss_fn, rows, epochs):
            opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            for ep in range(epochs):
                perm = torch.as_tensor(rng.permutation(rows), device=device)
                for i in range(0, len(perm), cfg.batch):
                    b = perm[i : i + cfg.batch]
                    loss = loss_fn(net, X[b], Y[b])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sched.step()
                if ep % 20 == 0 or ep == epochs - 1:
                    with torch.no_grad():
                        log(f"  epoch {ep:3d}  loss {loss_fn(net, X[rows], Y[rows]).item():.5f}")

        log("fitting mean model")
        train(self.mean_net, lambda n, x, y: ((n(x) - y) ** 2).mean(), tr, cfg.epochs)
        with torch.no_grad():
            R = Y - self.mean_net(X)

        log("fitting noise model on held-out residuals")
        Xv, Rv = X[va], R[va]
        opt = torch.optim.Adam(self.noise_net.parameters(), lr=cfg.lr)
        for ep in range(cfg.epochs):
            perm = torch.as_tensor(rng.permutation(len(Xv)), device=device)
            for i in range(0, len(perm), cfg.batch):
                b = perm[i : i + cfg.batch]
                s = self.sigma_floor + nn.functional.softplus(self.noise_net(Xv[b]))
                loss = (torch.log(s) + 0.5 * (Rv[b] / s) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            s = self.sigma_floor + nn.functional.softplus(self.noise_net(Xv))
            z = (Rv / s).cpu().numpy()
        log(f"  held-out residual rmse {Rv.pow(2).mean().sqrt().item():.4f}, standardized std {z.std(0).round(2)}")
        self.mean_net.cpu().eval()
        self.noise_net.cpu().eval()
        return self

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        torch.save(
            {
                "env_cfg": asdict(self.env_cfg),
                "mean": self.mean_net.state_dict(),
                "noise": self.noise_net.state_dict(),
                "hidden": self.mean_net[0].out_features,
                "sigma_floor": self.sigma_floor,
                "stats": {k: getattr(self, k) for k in ("z_mu", "z_sd", "a_scale", "d_sd")},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, env_cfg: EnvConfig | None = None) -> "FrozenModel":
        ck = torch.load(path, weights_only=False)
        m = cls(env_cfg or EnvConfig(**ck["env_cfg"]), ck["hidden"], ck["sigma_floor"])
        m.mean_net.load_state_dict(ck["mean"])
        m.noise_net.load_state_dict(ck["noise"])
        for k, v in ck["stats"].items():
            setattr(m, k, np.asarray(v))
        m.mean_net.eval()
        m.noise_net.eval()
        return m


def features(a_n, dfn, dfn0):
    """phi(z, a) = [a, f(z,a) - f(z,0), f(z,0) - z, 1] in normalized units (p = 11).

    The frozen prediction is split into its action-driven part and its passive part. A mass change
    rescales the action-driven part (weight m0/m - 1) and a friction change rescales the passive
    sliding part (weight mu/mu0 - 1), so the residual is linear in phi with O(1) weights
    (Assumption 1 in the paper). The raw action, scaled to the training mass, absorbs the friction
    acting during the impulse. The same split applies to a latent world model in stage 1b.
    """
    parts = [a_n, dfn - dfn0, dfn0]
    if isinstance(dfn, torch.Tensor):
        return torch.cat([*parts, torch.ones_like(dfn[..., :1])], -1)
    return np.concatenate([*parts, np.ones_like(dfn[..., :1])], -1)


K_RESIDUAL = 4
P_FEATURES = 2 + 2 * K_RESIDUAL + 1


def structured_prior_var(own: float = 1.0, cross: float = 0.01) -> np.ndarray:
    """(k, p) prior variances: each residual coordinate mainly rescales its own frozen change and the
    action on its axis (friction and mass changes), with small variance on all other weights."""
    pv = np.full((K_RESIDUAL, P_FEATURES), cross)
    for i in range(K_RESIDUAL):
        pv[i, i % 2] = own  # a_x -> x, vx ; a_y -> y, vy
        pv[i, 2 + i] = own  # action-driven change of coordinate i (mass)
        pv[i, 2 + K_RESIDUAL + i] = own  # passive change of coordinate i (friction)
    return pv


def episode_var_matrix(passive: float, action: float) -> np.ndarray:
    """(k, p) per-episode random-effect variance: friction jitters between episodes and rescales the
    passive change; the impulse only varies by execution noise. Mass does not jitter, so the
    action-driven weights get almost none; otherwise a memory could absorb a mass change within one
    strike step."""
    ev = np.zeros((K_RESIDUAL, P_FEATURES))
    for i in range(K_RESIDUAL):
        ev[i, 2 + K_RESIDUAL + i] = passive
        ev[i, 2 + i] = action
    return ev


def residual_table(model: FrozenModel, env_cfg: EnvConfig, episodes: int = 20, seed: int = 1) -> np.ndarray:
    """(surfaces, loads) mean |v| / sigma of velocity residuals on moderate strikes: how visible each
    context is to the frozen model (about 0.8 means indistinguishable from the training condition)."""
    env, rng = PuckEnv(env_cfg), np.random.default_rng(seed)
    out = np.zeros((len(env_cfg.surfaces), len(env_cfg.loads)))
    for s in range(len(env_cfg.surfaces)):
        for l in range(len(env_cfg.loads)):
            rows = []
            for _ in range(episodes):
                env.set_context(s, l, rng)
                env.reset((rng.uniform(*env_cfg.start_x), rng.uniform(*env_cfg.start_y)))
                r = env.strike(rng.uniform(env_cfg.action_low, env_cfg.action_high) * [0.4, 0.3], rng)
                if len(r.actions):
                    _, v, sig = model.residuals(r.states[:-1], r.actions, r.states[1:])
                    rows.append(np.abs(v / sig)[:, 2:])
            out[s, l] = np.concatenate(rows).mean()
    return out


def rollout_errors(model: FrozenModel, env_cfg: EnvConfig, n: int = 60, seed: int = 2) -> np.ndarray:
    """Open-loop final-position errors [m] of planner rollouts on the training condition."""
    from cira.config import PlannerConfig
    from cira.planner import CEMPlanner

    env, rng = PuckEnv(env_cfg), np.random.default_rng(seed)
    planner = CEMPlanner(model, env_cfg, PlannerConfig())
    errs = []
    for _ in range(n):
        env.set_context(*env_cfg.train_context)
        env.reset((rng.uniform(*env_cfg.start_x), rng.uniform(*env_cfg.start_y)))
        z0 = env.observe(None)
        a = rng.uniform([0.5, -0.2], [1.5, 0.2])
        r = env.strike(a, None, z0=z0)
        if not r.off_table:
            p = planner.rollout(z0, a[None], np.zeros((1, K_RESIDUAL, P_FEATURES)))[0, 0, -1].numpy()
            errs.append(np.linalg.norm(p - r.final_xy))
    return np.array(errs)
