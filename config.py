from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
ORIGINALS_DIR = ROOT / "data" / "originals"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

SEED = 42
IMG_SIZE = 96
VAL_FRACTION = 0.15
BATCH_SIZE = 32
EPOCHS = 12
DUPLICATE_DISTANCE = 6   # max Hamming distance (of 64 bits) between perceptual hashes of near-duplicates

# Quality costs: letting a defective part through is far worse than scrapping a good one
COST_ESCAPE = 20.0
COST_FALSE_REJECT = 1.0
TARGET_ESCAPE_RATE = 0.002   # triage: at most 0.2% of defective parts may be auto-accepted
TARGET_FALSE_REJECT = 0.01   # triage: at most 1% of good parts may be auto-rejected
