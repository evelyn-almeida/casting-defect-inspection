import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import IMG_SIZE, SEED


def handcrafted_features(images):
    """Classical machine vision: shape/edge histograms (HOG) + local texture (LBP)."""
    feats = []
    for img in images[..., 0]:
        h = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
        lbp = local_binary_pattern((img * 255).astype(np.uint8), P=8, R=1, method="uniform")
        hist = np.histogram(lbp, bins=10, range=(0, 10), density=True)[0]
        feats.append(np.concatenate([h, hist]))
    return np.array(feats)


def classical_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000, random_state=SEED))


def cnn(input_size=IMG_SIZE, robust=False):
    """Compact CNN trained from scratch. Rotation/flip augmentation: a casting looks the same rotated.

    robust=True adds stronger lighting changes and sensor noise to the augmentation.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    tf.random.set_seed(SEED)
    photometric = 0.3 if robust else 0.1
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.5, fill_mode="reflect", seed=SEED),
        layers.RandomBrightness(photometric, value_range=(0, 1), seed=SEED),
        layers.RandomContrast(2 * photometric, seed=SEED),
        *([layers.GaussianNoise(0.06, seed=SEED)] if robust else []),
    ], name="augmentation")

    inputs = layers.Input((input_size, input_size, 1))
    x = augment(inputs)
    for i, filters in enumerate([16, 32, 64, 128]):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.GroupNormalization(groups=8)(x)  # no running statistics: same behaviour in training and inference
        x = layers.Activation("relu", name="last_conv" if i == 3 else None)(x)
        if i < 3:
            x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="casting_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"])
    return model
