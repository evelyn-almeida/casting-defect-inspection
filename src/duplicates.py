"""Data audit: how similar is each test image to its closest training image?

Images are compared on their residual after removing the average casting (all parts share
the same shape), over the 8 rotations/flips, using normalized correlation.
"""
import numpy as np
from PIL import Image

SIZE = 64


def _load(paths):
    return np.stack([np.asarray(Image.open(p).convert("L").resize((SIZE, SIZE), Image.BILINEAR), float)
                     for p in paths])


def _dihedral(x):
    v = [np.rot90(x, k, axes=(-2, -1)) for k in range(4)]
    return np.stack(v + [np.flip(a, -1) for a in v])


class SimilarityIndex:
    def __init__(self, reference_paths):
        ref = _load(reference_paths)
        self.mean = _dihedral(ref[::20]).reshape(-1, SIZE, SIZE).mean(axis=0)
        self.ref = self._residual(ref)

    def _residual(self, x):
        r = (x - self.mean).reshape(len(x), -1)
        r -= r.mean(axis=1, keepdims=True)
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def nearest(self, paths, exclude_self=False):
        """Similarity and index of the closest reference image for each query."""
        sims, idx = np.empty(len(paths)), np.empty(len(paths), dtype=int)
        for i, x in enumerate(_load(paths)):
            s = (self._residual(_dihedral(x)) @ self.ref.T).max(axis=0)
            if exclude_self:
                s[i] = -1
            idx[i], sims[i] = s.argmax(), s.max()
        return sims, idx
