"""CIRA and baselines behind one interface.

Every method exposes its pre-contact knowledge as a list of hypotheses (probability, W), where W
(k x p) maps features phi(z, a) to the normalized residual correction. The planner rolls out the
frozen model plus each hypothesis's correction, so the methods differ only in how they choose and
learn W, never in the planner.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from cira.config import CiraConfig, SGDConfig
from cira.frozen import K_RESIDUAL as K
from cira.frozen import P_FEATURES as P
from cira.frozen import episode_var_matrix, structured_prior_var
from cira.memories import BLRMemory, CueModel, GlobalCue


def memory_prior(cfg: CiraConfig):
    if cfg.prior == "structured":
        return structured_prior_var(cfg.prior_var, cfg.prior_var_cross)
    return cfg.prior_var


def _normalize_log(lp: np.ndarray) -> np.ndarray:
    return lp - np.logaddexp.reduce(lp)


class Agent:
    name = "agent"

    def begin_episode(self, cue: np.ndarray, true_ctx: int | None = None) -> None:
        pass

    def hypotheses(self) -> list[tuple[float, np.ndarray]]:
        raise NotImplementedError

    def observe(self, phi: np.ndarray, v: np.ndarray, sigma: np.ndarray) -> None:
        pass

    def end_episode(self, true_ctx: int | None = None) -> None:
        pass

    def observe_episode(self, PHI, V, SIG) -> None:
        for phi, v, s in zip(PHI, V, SIG):
            self.observe(phi, v, s)

    def diagnostics(self) -> dict:
        return {}


class Frozen(Agent):
    name = "Frozen"

    def hypotheses(self):
        return [(1.0, np.zeros((K, P)))]


class SGDAdapter(Agent):
    """Residual adapter trained by normalized-LMS gradient steps on each transition.

    reset=True is the episode-reset family (AdaJEPA / Sandwich-Residuals style); reset=False is
    Continual-SGD. Same function class as a CIRA memory, so only the learning rule differs.
    """

    def __init__(self, cfg: SGDConfig, reset: bool):
        self.cfg, self.reset = cfg, reset
        self.name = "Reset-SGD" if reset else "Continual-SGD"
        self.W = np.zeros((K, P))

    def begin_episode(self, cue, true_ctx=None):
        if self.reset:
            self.W = np.zeros((K, P))

    def hypotheses(self):
        return [(1.0, self.W.copy())]

    def observe(self, phi, v, sigma):
        for _ in range(self.cfg.steps):
            err = v - self.W @ phi
            self.W += self.cfg.lr * err[:, None] * phi[None, :] / (phi @ phi + 1e-6)


class CueRLS(Agent):
    """Cue-conditioned continual regression: W(y) linear in [y, 1], fitted in closed form, no resets.

    The strongest simple competitor to cue-driven recall (image-to-property maps, RMA-style
    conditioning). It uses the cue but not the temporal structure of the stream.
    """

    name = "Cue-RLS"

    def __init__(self, cue_dim: int, prior_var: float = 4.0):
        self.q = cue_dim + 1
        self.reg = BLRMemory(K, P * self.q, prior_var)
        self.y = np.zeros(self.q)

    def begin_episode(self, cue, true_ctx=None):
        self.y = np.append(cue, 1.0)

    def _psi(self, phi):
        return (phi[:, None] * self.y[None, :]).ravel()

    def hypotheses(self):
        return [(1.0, np.einsum("kpq,q->kp", self.reg.W.reshape(K, P, self.q), self.y))]

    def observe(self, phi, v, sigma):
        self.reg.update(self._psi(phi), v, 1.0 / sigma**2)


class Oracle(Agent):
    """One memory per true context, selected and updated with the true label."""

    name = "Oracle"

    def __init__(self, prior_var: float = 4.0):
        self.prior_var = prior_var
        self.mems: dict[int, BLRMemory] = {}
        self.cur: BLRMemory | None = None

    def begin_episode(self, cue, true_ctx=None):
        self.cur = self.mems.setdefault(true_ctx, BLRMemory(K, P, self.prior_var))

    def hypotheses(self):
        return [(1.0, self.cur.W.copy())]

    def observe(self, phi, v, sigma):
        self.cur.update(phi, v, 1.0 / sigma**2)


class CIRA(Agent):
    """Context-Indexed Residual Adapters (paper Sec. 3), without the factored prior and safety filter.

    Hypotheses are the stored memories plus a new-context hypothesis (the last entry). Memory 0 is
    the fixed training condition. Two filters share the memories: the main filter (history x cue)
    decides, plans and weights updates; a parallel cue-free filter supplies the labels on which the
    cue models are trained, so a cue is never reinforced by its own predictions.

    Within an episode each hypothesis is scored by a per-episode copy of its memory ("scorer") that
    has seen every earlier step of the episode at full weight, plus a small random effect for
    episode-to-episode variation within a context. Because the context is fixed within an episode
    this is the exact sequential likelihood; without it the new-context hypothesis stays at its
    broad prior predictive and loses to any roughly matching memory. The library itself is updated
    with predictive weights. Memories carry stable ids so that merges keep logs interpretable.
    """

    NOVEL = -1

    def __init__(self, cfg: CiraConfig, cue_dim: int, name: str = "CIRA"):
        self.cfg, self.name, self.dim = cfg, name, cue_dim
        self.prior = memory_prior(cfg)
        self.episode_var = episode_var_matrix(cfg.episode_var, cfg.episode_var_action)
        self.mems = [BLRMemory(K, P, self.prior, fixed=True)]
        self.ids = [0]
        self.next_id = 1
        self.alias: dict[int, int] = {}  # merged id -> surviving id
        self.counts = np.array([cfg.train_pseudocount])
        self.cues = [CueModel(cue_dim, cfg.cue_pseudocount, cfg.cue_var_floor)]
        self.glob = GlobalCue(cue_dim, cfg.cue_var_floor)
        self.cand = self._fresh()
        self.prev = np.array([1.0])
        self.assign: dict[int, dict[int, float]] = {0: {}}  # evaluation only: belief mass per true context
        self.buffer: deque = deque(maxlen=cfg.merge_buffer)
        self.lq = self.lq_nc = np.zeros(2)
        self.pre_contact: dict[int, float] = {0: 1.0}
        self.created_this_episode = self.merged_this_episode = 0

    def _fresh(self) -> BLRMemory:
        return BLRMemory(K, P, self.prior)

    # ------------------------------------------------------------------ priors
    def _log_prior(self, cue, use_cue: bool) -> np.ndarray:
        c = self.cfg
        crp = np.append(self.counts, c.crp_alpha) / (self.counts.sum() + c.crp_alpha)
        lp = np.log(c.stay_prob * np.append(self.prev, 0.0) + (1.0 - c.stay_prob) * crp)
        if use_cue:
            gm, gv = self.glob.params()
            ll = [m.loglik(cue, gm, gv) for m in self.cues]
            ll.append(float(-0.5 * np.sum(np.log(2 * np.pi * gv) + (cue - gm) ** 2 / gv)))
            lp = lp + c.cue_temperature * np.array(ll)
        return _normalize_log(lp)

    def begin_episode(self, cue, true_ctx=None):
        self.cue = np.asarray(cue, float)
        self.lq = self._log_prior(self.cue, self.cfg.use_cue)
        self.lq_nc = self._log_prior(self.cue, False)
        p = np.exp(self.lq)
        self.pre_contact = {**dict(zip(self.ids, p[:-1])), self.NOVEL: p[-1]}
        self.scorers = [m.copy(self.episode_var) for m in self.mems] + [self.cand.copy()]
        self.streak = 0
        self.created_this_episode = self.merged_this_episode = 0

    def hypotheses(self):
        p = np.exp(self.lq)
        keep = [i for i in range(len(p)) if p[i] >= self.cfg.prune] or [int(np.argmax(p))]
        Ws = [m.W for m in self.mems] + [self.cand.W]
        z = p[keep].sum()
        return [(p[i] / z, Ws[i].copy()) for i in keep]

    # ------------------------------------------------------------------ filtering and learning
    def observe(self, phi, v, sigma):
        scorers = self.scorers if self.cfg.episode_conditional else self.mems + [self.cand]
        ll = np.array([m.loglik(phi, v, sigma) for m in scorers])
        q = np.exp(self.lq)
        lpost = _normalize_log(self.lq + ll)
        post = np.exp(lpost)
        if self.cfg.weights == "predictive":
            w = q
        elif self.cfg.weights == "filtered":
            w = post
        else:
            w = np.eye(len(post))[np.argmax(post)]
        inv_var = 1.0 / sigma**2
        for wi, m in zip(w[:-1], self.mems):
            m.update(phi, v, wi * inv_var)
        if not self.cfg.episode_conditional:
            self.cand.update(phi, v, w[-1] * inv_var)
        for m in self.scorers:
            m.update(phi, v, inv_var)
        self.buffer.append((phi, sigma))
        self.lq = lpost
        self.lq_nc = _normalize_log(self.lq_nc + ll)
        self.streak = self.streak + 1 if np.argmax(post) == len(self.mems) else 0
        if self.streak >= self.cfg.create_after:
            self._instantiate()

    def _instantiate(self) -> None:
        """Turn the new-context hypothesis into a memory; its belief mass moves with it.

        With episode-conditional scoring the new memory starts from the new-context scorer, i.e. the
        posterior given this episode's data under the hypothesis that just won. Starting from the
        q-weighted candidate instead leaves it near the prior (q of the new hypothesis is small until
        evidence accumulates), and short episodes then create duplicates of the same context.
        """
        if self.cfg.episode_conditional:
            self.mems.append(self.scorers[-1].copy())
        else:
            self.mems.append(self.cand)
            self.cand = self._fresh()
        self.scorers.append(self._fresh())  # the old new-context scorer now scores the new memory
        self.ids.append(self.next_id)
        self.assign[self.next_id] = {}
        self.next_id += 1
        self.counts = np.append(self.counts, 0.0)
        self.prev = np.append(self.prev, 0.0)
        self.cues.append(CueModel(self.dim, self.cfg.cue_pseudocount, self.cfg.cue_var_floor))
        eps = np.log(self.cfg.novel_reset_mass)
        self.lq = _normalize_log(np.append(self.lq, eps))
        self.lq_nc = _normalize_log(np.append(self.lq_nc, eps))
        self.streak = 0
        self.created_this_episode += 1

    def end_episode(self, true_ctx=None):
        post = np.exp(self.lq)
        if np.argmax(post) == len(self.mems):
            self._instantiate()  # short episodes (e.g. puck left the table) may end before the streak
            post = np.exp(self.lq)
        post_nc = np.exp(self.lq_nc)
        belief = post[:-1] / post[:-1].sum()
        self.counts += belief
        self.prev = belief
        if self.cfg.use_cue:
            label = post_nc[:-1] / post_nc[:-1].sum()
            for w, m in zip(label, self.cues):
                if w > 1e-4:
                    m.add(self.cue, w)
        self.glob.add(self.cue)
        if true_ctx is not None:
            for mid, w in zip(self.ids, belief):
                self.assign[mid][true_ctx] = self.assign[mid].get(true_ctx, 0.0) + w
        self.cand = self._fresh()
        if self.cfg.merge_z2 > 0:
            self._merge_duplicates()

    # ------------------------------------------------------------------ merging
    def _merge_duplicates(self) -> None:
        """Merge pairs of memories whose predictions agree within noise plus uncertainty on every
        recently visited feature vector (proposal: 'two memories that become nearly identical')."""
        if len(self.buffer) < 8 or len(self.mems) < 2:
            return
        Phi = np.stack([b[0] for b in self.buffer])
        S2 = np.stack([b[1] for b in self.buffer]) ** 2
        means = [Phi @ m.W.T for m in self.mems]
        var = [np.zeros_like(S2) if m.fixed else np.einsum("ni,kij,nj->nk", Phi, m.Vinv, Phi) for m in self.mems]
        best = None
        for i in range(len(self.mems)):
            for j in range(i + 1, len(self.mems)):
                z2 = (means[i] - means[j]) ** 2 / (S2 + var[i] + var[j])
                score = float(z2.max())
                if score < self.cfg.merge_z2 and (best is None or score < best[0]):
                    best = (score, i, j)
        if best is not None:
            self._merge(best[1], best[2])
            self._merge_duplicates()

    def _merge(self, i: int, j: int) -> None:
        """Fold memory j into memory i (i < j, so the fixed training memory always survives)."""
        if not self.mems[i].fixed:
            self.mems[i] = self.mems[i].merged_with(self.mems[j])
        self.counts[i] += self.counts[j]
        self.prev[i] += self.prev[j]
        self.cues[i].absorb(self.cues[j])
        keep, gone = self.ids[i], self.ids[j]
        for ctx, w in self.assign.pop(gone).items():
            self.assign[keep][ctx] = self.assign[keep].get(ctx, 0.0) + w
        self.alias[gone] = keep
        for lst in (self.mems, self.ids, self.cues):
            del lst[j]
        self.counts = np.delete(self.counts, j)
        self.prev = np.delete(self.prev, j)
        self.merged_this_episode += 1

    # ------------------------------------------------------------------ evaluation helpers
    def resolve(self, mid: int) -> int:
        while mid in self.alias:
            mid = self.alias[mid]
        return mid

    def memory_labels(self) -> dict[int, int]:
        """Majority true context per memory id, merged ids included (evaluation only)."""
        labels = {mid: (max(a, key=a.get) if a else -1) for mid, a in self.assign.items()}
        return {mid: labels[self.resolve(mid)] for mid in list(self.assign) + list(self.alias)}

    def diagnostics(self):
        return {"n_memories": len(self.mems), "created": self.created_this_episode, "merged": self.merged_this_episode}


def make_agent(name: str, cira_cfg: CiraConfig, sgd_cfg: SGDConfig, cue_dim: int) -> Agent:
    if name == "Frozen":
        return Frozen()
    if name == "Reset-SGD":
        return SGDAdapter(sgd_cfg, reset=True)
    if name == "Continual-SGD":
        return SGDAdapter(sgd_cfg, reset=False)
    if name == "Cue-RLS":
        return CueRLS(cue_dim, cira_cfg.prior_var)
    if name == "Oracle":
        return Oracle(memory_prior(cira_cfg))
    if name == "Mixture":  # MOLe-style: error-driven selection, plain CRP, filtered weights
        return CIRA(replace(cira_cfg, use_cue=False, stay_prob=0.0, weights="filtered"), cue_dim, name)
    if name == "CIRA-NoCue":
        return CIRA(replace(cira_cfg, use_cue=False), cue_dim, name)
    if name == "CIRA-Filtered":
        return CIRA(replace(cira_cfg, weights="filtered"), cue_dim, name)
    if name == "CIRA":
        return CIRA(cira_cfg, cue_dim, name)
    raise ValueError(f"unknown method {name}")


METHODS = ["Frozen", "Reset-SGD", "Continual-SGD", "Cue-RLS", "Mixture", "CIRA-NoCue", "CIRA", "Oracle"]
