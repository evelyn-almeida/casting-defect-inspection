import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from config import COST_ESCAPE, COST_FALSE_REJECT, FIG_DIR

BLUE, ORANGE, GREEN, RED, GRAY = "#1f4e79", "#e07a1f", "#2a9d8f", "#c0392b", "#8a8a8a"
MODEL_COLORS = {"HOG + LBP + logistic regression": ORANGE, "CNN (standard augmentation)": BLUE,
                "CNN (+ noise & lighting augmentation)": GREEN}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _gallery(axes, images, titles, colors=None):
    for i, ax in enumerate(np.ravel(axes)):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i][..., 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(titles[i], fontsize=9, color=(colors or [None] * len(images))[i] or "black")


def samples(ok, defective):
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
    _gallery(axes[0], ok, ["OK"] * 6, [GREEN] * 6)
    _gallery(axes[1], defective, ["Defective"] * 6, [RED] * 6)
    fig.suptitle("Submersible pump impellers, top view: blowholes, pinholes, burrs, shrinkage and edge defects",
                 fontsize=13, fontweight="bold")
    save(fig, "01_samples")


def data_audit(sims, pairs):
    fig = plt.figure(figsize=(15, 5))
    grid = fig.add_gridspec(2, 5, width_ratios=[2.6, 1, 1, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    bins = np.linspace(0.7, 1, 61)
    for (label, s), color in zip(sims.items(), [RED, BLUE, GRAY]):
        ax.hist(s, bins=bins, density=True, alpha=0.5, color=color, label=f"{label} (median {np.median(s):.3f})")
    ax.set(xlabel="Similarity to the closest other image", ylabel="Density")
    ax.legend(loc="upper left", fontsize=8)
    for k, (test_img, train_img, s) in enumerate(pairs[:4]):
        for row, (img, name) in enumerate([(test_img, "test"), (train_img, "closest train")]):
            a = fig.add_subplot(grid[row, k + 1])
            a.imshow(img, cmap="gray")
            a.set_title(f"{name}" + (f"\nsim = {s:.3f}" if row == 0 else ""), fontsize=8)
            a.axis("off")
    fig.suptitle("Data audit: test images are closer to training images than distinct parts are to each other",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "02_data_audit")


def training_curves(histories):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for name, h in histories.items():
        epochs = h.index + 1
        for ax, metric in [(axes[0], "loss"), (axes[1], "auc")]:
            ax.plot(epochs, h[metric], color=MODEL_COLORS[name], lw=1.5, ls="--", alpha=0.7)
            ax.plot(epochs, h[f"val_{metric}"], color=MODEL_COLORS[name], lw=2.5, marker="o", label=name)
    axes[0].set(xlabel="Epoch", title="Binary cross-entropy (dashed: training, solid: validation)", yscale="log")
    axes[1].set(xlabel="Epoch", title="ROC AUC", ylim=(0.8, 1.005))
    axes[1].legend(loc="lower right")
    fig.suptitle("CNN training", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_training_curves")


def roc_pr(y, scores):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, lw=2.5, color=MODEL_COLORS[name], label=f"{name} (AUC = {auc(fpr, tpr):.4f})")
        axes[1].plot(rec, prec, lw=2.5, color=MODEL_COLORS[name], label=name)
    axes[0].plot([0, 1], [0, 1], color=GRAY, ls=":")
    axes[0].set(xlabel="Good parts rejected (false positive rate)", ylabel="Defects caught (recall)",
                title="ROC curve")
    axes[1].set(xlabel="Defects caught (recall)", ylabel="Precision", title="Precision-recall curve", ylim=(0.6, 1.01))
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "04_roc_pr")


def confusion(matrices):
    fig, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 4.8))
    for ax, (title, m) in zip(axes, matrices.items()):
        ax.imshow(m, cmap="Blues")
        labels = ["OK", "Defective"]
        ax.set_xticks([0, 1], [f"Predicted {l}" for l in labels])
        ax.set_yticks([0, 1], [f"True {l}" for l in labels])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=16, fontweight="bold",
                        color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_title(title)
        ax.grid(False)
    fig.suptitle("Test set (715 parts) at the cost-optimal threshold chosen on validation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "05_confusion_matrices")


def cost_threshold(thresholds, costs, chosen):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, cost in costs.items():
        ax.plot(thresholds, cost, lw=2.5, color=MODEL_COLORS[name], label=name)
        ax.axvline(chosen[name], color=MODEL_COLORS[name], ls=":", lw=1.5)
    ax.set(xlabel="Decision threshold on P(defective)", ylabel="Expected cost per part (test set)",
           title=f"Cost-sensitive threshold: a missed defect costs {COST_ESCAPE:g}× a scrapped good part",
           yscale="symlog", ylim=(0, None))
    ax.legend(loc="upper center")
    save(fig, "06_cost_threshold")


def triage(composition, summary):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = {"Auto-accept": GREEN, "Human review": ORANGE, "Auto-reject": RED}
    left = np.zeros(len(composition))
    for col, color in colors.items():
        ax.barh(composition.index, composition[col], left=left, color=color, label=col)
        for i, v in enumerate(composition[col]):
            if v > 0.03:
                ax.text(left[i] + v / 2, i, f"{v:.1%}", ha="center", va="center", color="white", fontweight="bold")
        left += composition[col].to_numpy()
    ax.set(xlim=(0, 1), xlabel="Share of parts", title=summary)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    save(fig, "07_triage")


def grad_cam(images, cams, probs):
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    for i in range(n):
        axes[0, i].imshow(images[i][..., 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"P(defect) = {probs[i]:.2f}", fontsize=9)
        axes[1, i].imshow(images[i][..., 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(cams[i], cmap="jet", alpha=0.45)
        for a in axes[:, i]:
            a.axis("off")
    axes[0, 0].text(-0.15, 0.5, "Image", transform=axes[0, 0].transAxes, rotation=90, va="center", fontsize=11)
    axes[1, 0].text(-0.15, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes, rotation=90, va="center", fontsize=11)
    fig.suptitle("Where the CNN looks: heatmaps highlight the defect regions", fontsize=13, fontweight="bold")
    save(fig, "08_grad_cam")


def errors(images, titles, total):
    n = max(len(images), 1)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2), squeeze=False)
    _gallery(axes[0], images, titles, [RED] * len(images))
    fig.suptitle(f"Mistakes of the deployed CNN on the test set ({len(images)} of {total} shown)" if total else
                 "The deployed CNN made no mistakes on the test set", fontsize=13, fontweight="bold")
    save(fig, "09_errors")


def robustness(results):
    names = list(results["perturbation"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), sharey=True)
    for ax, name in zip(axes, names):
        g = results[results["perturbation"] == name]
        for model, m in g.groupby("model", sort=False):
            ax.plot(m["level"], m["accuracy"], marker="o", lw=2.5, color=MODEL_COLORS[model], label=model)
        ax.set(title=name, xlabel="Severity")
    axes[0].set(ylabel="Accuracy (validation threshold)", ylim=(0.4, 1.02))
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Robustness to camera and lighting changes (never seen in training)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "10_robustness")
