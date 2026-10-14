"""Download the 'casting product image data for quality inspection' dataset (Dabhi, Kaggle).

Fetched from a public GitHub mirror with a sparse git checkout (requires git).
The same data is available on Kaggle: ravirajsinh45/real-life-industrial-dataset-of-casting-product
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, ORIGINALS_DIR

MIRROR = "https://github.com/kwakjm/casting_products"
SUBDIR = "data/Casting material/casting_data/casting_data"
ORIGINALS = "data/Casting material/casting_512x512/casting_512x512"   # 1,300 un-augmented photos, for the data audit


def main():
    if (DATA_DIR / "train").exists():
        print(f"data already in {DATA_DIR}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        run = lambda *args: subprocess.run(["git", *args], cwd=tmp, check=True)
        run("clone", "--depth", "1", "--filter=blob:none", "--sparse", MIRROR, "repo")
        subprocess.run(["git", "sparse-checkout", "set", "--no-cone", SUBDIR, ORIGINALS],
                       cwd=Path(tmp) / "repo", check=True)
        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(tmp) / "repo" / SUBDIR, DATA_DIR)
        shutil.copytree(Path(tmp) / "repo" / ORIGINALS, ORIGINALS_DIR)
    print(f"data ready in {DATA_DIR}")


if __name__ == "__main__":
    main()
