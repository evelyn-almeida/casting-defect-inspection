"""Image degradations a camera on a production line can produce."""
import numpy as np
from scipy.ndimage import gaussian_filter

PERTURBATIONS = {
    "Sensor noise (σ)": ([0, 0.02, 0.05, 0.08, 0.12], lambda x, s, rng: x + rng.normal(0, s, x.shape)),
    "Defocus blur (σ, px)": ([0, 0.5, 1.0, 1.5, 2.0],
                             lambda x, s, rng: gaussian_filter(x, (0, s, s, 0)) if s else x),
    "Lighting: brighter (+)": ([0, 0.05, 0.1, 0.2, 0.3], lambda x, s, rng: x + s),
    "Lighting: lower contrast (×)": ([1, 0.85, 0.7, 0.55, 0.4], lambda x, s, rng: 0.5 + (x - 0.5) * s),
}


def apply(images, name, level, seed=0):
    fn = PERTURBATIONS[name][1]
    return np.clip(fn(images, level, np.random.default_rng(seed)), 0, 1).astype(np.float32)
