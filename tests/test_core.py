import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.decision import ACCEPT, REJECT, REVIEW, cost_optimal_threshold, expected_cost, triage, triage_thresholds
from src.models import handcrafted_features
from src.perturb import PERTURBATIONS, apply


def test_cost_prefers_low_thresholds_when_escapes_are_expensive():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.45, 0.4, 0.9])
    t = cost_optimal_threshold(y, p)
    assert t <= 0.4                        # catching the defect at 0.4 is worth one false reject
    cost, escapes, false_rejects = expected_cost(y, p, [0.5])
    assert escapes[0] == 1 and false_rejects[0] == 0


def test_triage_bands():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    p = np.r_[rng.beta(2, 8, 500), rng.beta(8, 2, 500)]
    t_low, t_high = triage_thresholds(y, p, max_escape=0.01, max_false_reject=0.01)
    d = triage(p, t_low, t_high)
    assert t_low <= t_high
    assert np.mean(d[y == 1] == ACCEPT) <= 0.01
    assert np.mean(d[y == 0] == REJECT) <= 0.011
    assert (d == REVIEW).any()


def test_perturbations_keep_valid_range():
    x = np.random.default_rng(0).random((2, 32, 32, 1)).astype(np.float32)
    for name, (levels, _) in PERTURBATIONS.items():
        out = apply(x, name, levels[-1])
        assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
        assert np.allclose(apply(x, name, levels[0]), x, atol=1e-6)


def test_handcrafted_features_shape():
    x = np.random.default_rng(0).random((3, 96, 96, 1)).astype(np.float32)
    f = handcrafted_features(x)
    assert f.shape[0] == 3 and np.isfinite(f).all()
