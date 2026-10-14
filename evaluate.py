"""Evaluate the trained CNN against a classical baseline and produce all figures: python evaluate.py"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

from config import (MODEL_DIR, ORIGINALS_DIR, RESULTS_DIR, SEED, TARGET_ESCAPE_RATE, TARGET_FALSE_REJECT)
from src import plots
from src.data import index, load_images
from src.decision import cost_optimal_threshold, expected_cost, triage, triage_thresholds
from src.duplicates import SimilarityIndex
from src.explain import grad_cam
from src.models import classical_model, handcrafted_features
from src.perturb import PERTURBATIONS, apply
from train import split_train_val

CLASSICAL = "HOG + LBP + logistic regression"
CNN, ROBUST = "CNN (standard augmentation)", "CNN (+ noise & lighting augmentation)"


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    return {"ROC AUC": roc_auc_score(y, p), "Accuracy": accuracy_score(y, pred),
            "Defect recall": recall_score(y, pred), "Good-part recall": recall_score(1 - y, 1 - pred),
            "F1": f1_score(y, pred), "Threshold": t}


def main():
    df = index()
    fit, val = split_train_val(df)
    test = df[df["split"] == "test"]
    X_fit, X_val, X_test = (load_images(df.loc[i, "path"]) for i in (fit, val, test.index))
    y_fit, y_val, y_test = (df.loc[i, "label"].to_numpy() for i in (fit, val, test.index))

    cnn = tf.keras.models.load_model(MODEL_DIR / "cnn.keras")
    robust = tf.keras.models.load_model(MODEL_DIR / "cnn_robust.keras")
    classical = classical_model().fit(handcrafted_features(X_fit), y_fit)
    predict = {CLASSICAL: lambda X: classical.predict_proba(handcrafted_features(X))[:, 1],
               CNN: lambda X: cnn.predict(X, verbose=0, batch_size=128)[:, 0],
               ROBUST: lambda X: robust.predict(X, verbose=0, batch_size=128)[:, 0]}
    p_val = {name: f(X_val) for name, f in predict.items()}
    p_test = {name: f(X_test) for name, f in predict.items()}
    thresholds = {name: cost_optimal_threshold(y_val, p) for name, p in p_val.items()}

    table = pd.DataFrame({name: metrics(y_test, p_test[name], thresholds[name]) for name in predict}).T
    grid = np.linspace(0.01, 0.99, 99)
    costs = {name: expected_cost(y_test, p, grid)[0] for name, p in p_test.items()}

    t_low, t_high = triage_thresholds(y_val, p_val[CNN], TARGET_ESCAPE_RATE, TARGET_FALSE_REJECT)
    decision = triage(p_test[CNN], t_low, t_high)
    names = ["Auto-accept", "Human review", "Auto-reject"]
    composition = pd.DataFrame({cls: [np.mean(decision[y_test == lab] == k) for k in range(3)]
                                for cls, lab in [("Defective parts", 1), ("Good parts", 0)]}, index=names).T
    triage_summary = {"automation_rate": float(np.mean(decision != 1)),
                      "escapes": int(np.sum((decision == 0) & (y_test == 1))),
                      "false_rejects": int(np.sum((decision == 2) & (y_test == 0))),
                      "t_low": t_low, "t_high": t_high}

    rows = []
    for name in PERTURBATIONS:
        for level in PERTURBATIONS[name][0]:
            Xp = apply(X_test, name, level, SEED)
            for model, f in predict.items():
                rows.append({"perturbation": name, "level": level, "model": model,
                             "accuracy": accuracy_score(y_test, f(Xp) >= thresholds[model])})
    robustness = pd.DataFrame(rows)

    train_paths = df.loc[fit.union(val), "path"].tolist()
    sim = SimilarityIndex(train_paths)
    s_test, nn_idx = sim.nearest(test["path"].tolist())
    sample = np.random.default_rng(SEED).choice(len(train_paths), 500, replace=False)
    s_train = np.sort(sim.ref[sample] @ sim.ref.T, axis=1)[:, -2]      # closest image other than itself
    originals = sorted(ORIGINALS_DIR.glob("*/*.jpeg"))
    s_orig = None
    if originals:
        orig = SimilarityIndex(originals)
        s_orig = np.sort(orig.ref[::3] @ orig.ref.T, axis=1)[:, -2]
    sims = {"Test → closest train": s_test, "Train → closest other train": s_train}
    if s_orig is not None:
        sims["Original photo → closest other original"] = s_orig
    order = np.argsort(s_test)[len(s_test) // 2: len(s_test) // 2 + 4]
    pairs = [(np.asarray(Image.open(test["path"].iloc[i]).convert("L")),
              np.asarray(Image.open(train_paths[nn_idx[i]]).convert("L")), s_test[i]) for i in order]
    audit = {"median_test_to_train": float(np.median(s_test)), "median_train_to_train": float(np.median(s_train)),
             "median_original_to_original": float(np.median(s_orig)) if s_orig is not None else None}

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "test_metrics.md").write_text(table.round(4).to_markdown())
    robustness.to_csv(RESULTS_DIR / "robustness.csv", index=False, float_format="%.4f")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(
        {"test_metrics": table.round(4).to_dict(orient="index"), "triage": triage_summary,
         "data_audit": audit}, indent=2))

    plots.setup_style()
    rng = np.random.default_rng(1)
    ok_idx = rng.choice(np.flatnonzero(y_test == 0), 6, replace=False)
    def_idx = rng.choice(np.flatnonzero(y_test == 1), 6, replace=False)
    plots.samples(X_test[ok_idx], X_test[def_idx])
    plots.data_audit(sims, pairs)
    plots.training_curves({CNN: pd.read_csv(RESULTS_DIR / "training_history.csv", index_col=0),
                           ROBUST: pd.read_csv(RESULTS_DIR / "training_history_robust.csv", index_col=0)})
    plots.roc_pr(y_test, p_test)
    plots.confusion({name: confusion_matrix(y_test, p_test[name] >= thresholds[name]) for name in predict})
    plots.cost_threshold(grid, costs, thresholds)
    plots.triage(composition, f"Triage on the test set: {triage_summary['automation_rate']:.1%} of parts decided "
                              f"automatically, {triage_summary['escapes']} defects let through")
    shown = np.flatnonzero(y_test == 1)[:: max(1, int(np.sum(y_test == 1) / 7))][:7]
    plots.grad_cam(X_test[shown], grad_cam(cnn, X_test[shown]), p_test[CNN][shown])
    wrong = np.flatnonzero((p_test[CNN] >= thresholds[CNN]) != y_test)
    plots.errors(X_test[wrong][:8], [f"true: {'Defective' if y_test[i] else 'OK'}\nP(defect) = {p_test[CNN][i]:.2f}"
                                    for i in wrong[:8]], len(wrong))
    plots.robustness(robustness)

    print(table.round(4).to_string())
    print(json.dumps({"triage": triage_summary, "data_audit": audit}, indent=2))
    print(robustness.pivot_table(index=["perturbation", "level"], columns="model", values="accuracy").round(3))


if __name__ == "__main__":
    main()
