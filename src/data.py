import numpy as np
import pandas as pd
from PIL import Image

from config import DATA_DIR, IMG_SIZE

CLASS_NAMES = {0: "OK", 1: "Defective"}


def index():
    rows = [{"path": p, "split": split, "label": int(folder == "def_front")}
            for split in ["train", "test"] for folder in ["ok_front", "def_front"]
            for p in sorted((DATA_DIR / split / folder).glob("*.jpeg"))]
    return pd.DataFrame(rows)


def load_images(paths, size=IMG_SIZE):
    """Grayscale images in [0, 1], shape (n, size, size, 1)."""
    out = np.empty((len(paths), size, size, 1), dtype=np.float32)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
        out[i, ..., 0] = np.asarray(img, dtype=np.float32) / 255
    return out
