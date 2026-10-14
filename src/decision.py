"""Turning scores into accept / reject decisions with explicit quality costs."""
import numpy as np

from config import COST_ESCAPE, COST_FALSE_REJECT

ACCEPT, REVIEW, REJECT = 0, 1, 2


def expected_cost(y, p, thresholds):
    y, p = np.asarray(y), np.asarray(p)
    escapes = np.array([np.sum((p < t) & (y == 1)) for t in thresholds])
    false_rejects = np.array([np.sum((p >= t) & (y == 0)) for t in thresholds])
    return (COST_ESCAPE * escapes + COST_FALSE_REJECT * false_rejects) / len(y), escapes, false_rejects


def cost_optimal_threshold(y, p, thresholds=np.linspace(0.01, 0.99, 99)):
    cost, _, _ = expected_cost(y, p, thresholds)
    return float(thresholds[np.argmin(cost)])


def triage_thresholds(y, p, max_escape, max_false_reject):
    """Auto-accept below t_low, auto-reject above t_high, send the rest to a human."""
    y, p = np.asarray(y), np.asarray(p)
    t_low = float(np.quantile(p[y == 1], max_escape, method="lower"))
    t_high = float(np.quantile(p[y == 0], 1 - max_false_reject, method="higher"))
    return t_low, max(t_low, t_high)


def triage(p, t_low, t_high):
    return np.where(p < t_low, ACCEPT, np.where(p > t_high, REJECT, REVIEW))
