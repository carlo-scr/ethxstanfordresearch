"""Bayesian linear residual memories (paper eq. 1 and 4) and Gaussian cue models."""

from __future__ import annotations

import numpy as np

LOG2PI = np.log(2 * np.pi)


class BLRMemory:
    """k independent Bayesian linear regressions v_i ~ N(w_i^T phi, sigma_i^2) sharing one prior.

    Updates take a per-coordinate weight omega_i = q(c) / sigma_i^2, so the same class implements
    predictive, filtered and hard assignments. Inverses are maintained by Sherman-Morrison.
    A `fixed` memory keeps its prior mean (used for the training condition, mu_0 = 0).
    """

    def __init__(self, k: int, p: int, prior_var=4.0, prior_mean: np.ndarray | None = None, fixed: bool = False):
        """prior_var: scalar, or a (k, p) array of per-weight prior variances (diagonal prior)."""
        self.k, self.p, self.fixed = k, p, fixed
        nu = np.zeros((k, p)) if prior_mean is None else np.asarray(prior_mean, float).copy()
        pv = np.broadcast_to(np.asarray(prior_var, float), (k, p))
        self.Vinv = np.stack([np.diag(row) for row in pv])
        self.b = nu / pv  # P nu with P = diag(1 / prior_var)
        self.W = nu
        self.P0, self.b0 = 1.0 / pv, self.b.copy()  # prior information, subtracted when merging

    def copy(self, extra_var=None) -> "BLRMemory":
        """Copy; extra_var (k, p) adds a per-episode random effect to the weight covariance."""
        new = object.__new__(BLRMemory)
        new.k, new.p = self.k, self.p
        new.Vinv, new.W = self.Vinv.copy(), self.W.copy()
        new.P0, new.b0 = self.P0, self.b0
        new.fixed = self.fixed and extra_var is None
        if extra_var is not None:
            extra = np.stack([np.diag(row) for row in np.broadcast_to(extra_var, (self.k, self.p))])
            # A fixed memory has no parameter uncertainty, only the random effect (possibly singular).
            new.Vinv = extra if self.fixed else new.Vinv + extra
            new.b = None  # information form undefined; scorers are never merged
        else:
            new.b = self.b.copy()
        return new

    def merged_with(self, other: "BLRMemory") -> "BLRMemory":
        """Posterior given both memories' data: add their information and count the prior once."""
        prec = np.linalg.inv(self.Vinv) + np.linalg.inv(other.Vinv) - np.stack([np.diag(r) for r in self.P0])
        new = self.copy()
        new.Vinv = np.linalg.inv(prec)
        new.b = self.b + other.b - self.b0
        new.W = np.einsum("kij,kj->ki", new.Vinv, new.b)
        return new

    def predictive(self, phi: np.ndarray, sigma: np.ndarray):
        mean = self.W @ phi
        var = sigma**2 if self.fixed else sigma**2 + np.einsum("i,kij,j->k", phi, self.Vinv, phi)
        return mean, var

    def loglik(self, phi, v, sigma) -> float:
        mean, var = self.predictive(phi, sigma)
        return float(-0.5 * np.sum(LOG2PI + np.log(var) + (v - mean) ** 2 / var))

    def update(self, phi, v, omega) -> None:
        if self.fixed or np.max(omega) < 1e-10:
            return
        # Kalman form of the weighted Bayesian update (works with singular covariances).
        Vphi = self.Vinv @ phi  # (k, p)
        denom = 1.0 + omega * (Vphi @ phi)  # (k,)
        gain = (omega / denom)[:, None] * Vphi
        self.W = self.W + gain * (v - self.W @ phi)[:, None]
        self.Vinv -= gain[:, :, None] * Vphi[:, None, :]
        if self.b is not None:
            self.b += (omega * v)[:, None] * phi[None, :]


class CueModel:
    """Diagonal Gaussian over the cue, from weighted sufficient statistics shrunk to a global model."""

    def __init__(self, dim: int, pseudocount: float = 1.0, var_floor: float = 0.02):
        self.s0, self.s1, self.s2 = 0.0, np.zeros(dim), np.zeros(dim)
        self.tau, self.floor = pseudocount, var_floor

    def add(self, y, w: float = 1.0) -> None:
        self.s0 += w
        self.s1 += w * y
        self.s2 += w * y * y

    def absorb(self, other: "CueModel") -> None:
        self.s0 += other.s0
        self.s1 += other.s1
        self.s2 += other.s2

    def moments(self, prior_mean, prior_var):
        n = self.tau + self.s0
        mean = (self.tau * prior_mean + self.s1) / n
        ey2 = (self.tau * (prior_var + prior_mean**2) + self.s2) / n
        return mean, np.maximum(ey2 - mean**2, self.floor)

    def loglik(self, y, prior_mean, prior_var) -> float:
        mean, var = self.moments(prior_mean, prior_var)
        return float(-0.5 * np.sum(LOG2PI + np.log(var) + (y - mean) ** 2 / var))


class GlobalCue(CueModel):
    """Unweighted model of all cues seen so far; the cue model of the new-context hypothesis."""

    def __init__(self, dim: int, var_floor: float = 0.02):
        super().__init__(dim, pseudocount=1.0, var_floor=var_floor)
        self._m0, self._v0 = np.zeros(dim), np.ones(dim)

    def params(self):
        return self.moments(self._m0, self._v0)
