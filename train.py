"""Train the CNNs: python train.py  and  python train.py --robust  (about 10 min each on one CPU core)"""
import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import BATCH_SIZE, EPOCHS, MODEL_DIR, RESULTS_DIR, SEED, VAL_FRACTION
from src.data import index, load_images
from src.models import cnn


def split_train_val(df):
    train = df[df["split"] == "train"]
    fit, val = train_test_split(train.index, test_size=VAL_FRACTION, stratify=train["label"], random_state=SEED)
    return fit, val


def main(robust=False):
    suffix = "_robust" if robust else ""
    df = index()
    fit, val = split_train_val(df)
    X_fit, X_val = load_images(df.loc[fit, "path"]), load_images(df.loc[val, "path"])
    y_fit, y_val = df.loc[fit, "label"].to_numpy(), df.loc[val, "label"].to_numpy()

    model = cnn(robust=robust)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    start = time.time()
    history = model.fit(X_fit, y_fit, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=callbacks, verbose=2)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / f"cnn{suffix}.keras")
    pd.DataFrame(history.history).rename_axis("epoch").to_csv(RESULTS_DIR / f"training_history{suffix}.csv")
    print(f"trained in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robust", action="store_true", help="add sensor noise and stronger lighting augmentation")
    np.random.seed(SEED)
    main(parser.parse_args().robust)
