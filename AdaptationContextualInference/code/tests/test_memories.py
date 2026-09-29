import numpy as np

from cira.memories import BLRMemory, CueModel


def test_blr_matches_closed_form_ridge():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    W = rng.normal(size=(2, 5))
    Y = X @ W.T + 0.1 * rng.normal(size=(50, 2))
    m = BLRMemory(2, 5, prior_var=2.0)
    for x, y in zip(X, Y):
        m.update(x, y, np.full(2, 1 / 0.01))
    ridge = np.linalg.solve(X.T @ X / 0.01 + np.eye(5) / 2.0, X.T @ Y / 0.01).T
    assert np.allclose(m.W, ridge, atol=1e-8)
    assert np.allclose(m.Vinv[0], np.linalg.inv(X.T @ X / 0.01 + np.eye(5) / 2.0), atol=1e-8)


def test_merge_equals_joint_fit():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(40, 4)), rng.normal(size=(40, 1))
    a, b, joint = (BLRMemory(1, 4, 1.0) for _ in range(3))
    for i, (x, y) in enumerate(zip(X, Y)):
        (a if i < 20 else b).update(x, y, np.ones(1))
        joint.update(x, y, np.ones(1))
    assert np.allclose(a.merged_with(b).W, joint.W, atol=1e-8)


def test_random_effect_copy_keeps_mean():
    m = BLRMemory(2, 3, 1.0)
    m.update(np.ones(3), np.array([1.0, -1.0]), np.ones(2))
    c = m.copy(np.full((2, 3), 0.1))
    assert np.allclose(c.W, m.W) and np.all(np.diagonal(c.Vinv, axis1=1, axis2=2) > np.diagonal(m.Vinv, axis1=1, axis2=2))


def test_cue_model_prefers_its_own_mean():
    cm = CueModel(2)
    for _ in range(20):
        cm.add(np.array([2.0, 2.0]))
    gm, gv = np.zeros(2), np.ones(2)
    assert cm.loglik(np.array([2.0, 2.0]), gm, gv) > cm.loglik(np.array([-2.0, -2.0]), gm, gv)
