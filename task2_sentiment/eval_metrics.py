"""
Shared Task 2 evaluation module. Every member scores every model with this
file so numbers are directly comparable across the team.

    from eval_metrics import full_report, mcnemar_test, save_plots
    python eval_metrics.py --pred path/to/predictions_<model>.npz   # re-score saved predictions
"""
import argparse
import json

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Top-label ECE: confidence = max(p, 1-p), accuracy of the predicted label."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pred = (y_prob >= 0.5).astype(int)
    conf = np.where(pred == 1, y_prob, 1 - y_prob)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95, seed=1337):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = [metric_fn(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return [float(np.percentile(stats, (1 - ci) / 2 * 100)), float(np.percentile(stats, (1 + ci) / 2 * 100))]


def mcnemar_test(y_true, pred_a, pred_b):
    """Paired McNemar test on the discordant pairs. Exact binomial when b+c < 25,
    otherwise chi-squared with continuity correction."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))  # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))  # A wrong, B right
    if b + c == 0:
        return {"b_A_right_B_wrong": b, "c_A_wrong_B_right": c, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if b + c < 25:
        p = binomtest(b, b + c, 0.5).pvalue
        return {"b_A_right_B_wrong": b, "c_A_wrong_B_right": c, "statistic": float(min(b, c)), "p_value": float(p), "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {
        "b_A_right_B_wrong": b,
        "c_A_wrong_B_right": c,
        "statistic": float(stat),
        "p_value": float(1 - chi2.cdf(stat, df=1)),
        "method": "chi2_cc",
    }


def full_report(y_true, y_pred, y_prob, slices: dict = None) -> dict:
    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    r = {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "micro", "weighted"):
        r[f"precision_{avg}"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        r[f"recall_{avg}"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        r[f"f1_{avg}"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    r["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()  # rows=true [neg,pos]
    r["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    r["pr_auc"] = float(average_precision_score(y_true, y_prob))
    r["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    r["brier_score"] = float(brier_score_loss(y_true, y_prob))
    r["ece"] = expected_calibration_error(y_true, y_prob)
    r["error_rate"] = 1 - r["accuracy"]
    r["accuracy_ci95"] = bootstrap_ci(y_true, y_pred, accuracy_score)
    r["macro_f1_ci95"] = bootstrap_ci(y_true, y_pred, lambda t, p: f1_score(t, p, average="macro", zero_division=0))
    r["mcc_ci95"] = bootstrap_ci(y_true, y_pred, matthews_corrcoef)
    if slices:
        r["slices"] = {}
        for name, mask in slices.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            r["slices"][name] = {
                "n": int(mask.sum()),
                "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)),
                "error_rate": float((y_true[mask] != y_pred[mask]).mean()),
            }
    return r


def save_plots(results: dict, out_prefix: str):
    """results: {model_name: (y_true, y_prob)} -> ROC, PR and reliability diagrams."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, (y, p) in results.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y, p):.3f})")
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
        axes[2].plot(xs, ys, marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    for ax, (t, xl, yl) in zip(
        axes,
        [("ROC", "FPR", "TPR"), ("Precision-Recall", "Recall", "Precision"), ("Reliability", "mean predicted P(pos)", "fraction positive")],
    ):
        ax.set_title(t)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_curves.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (name, (y, p)) in zip(axes, results.items()):
        cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
        ax.set_xticks([0, 1], ["pred neg", "pred pos"])
        ax.set_yticks([0, 1], ["true neg", "true pos"])
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_confusion.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="npz with y_true, y_prob")
    a = ap.parse_args()
    d = np.load(a.pred)
    print(json.dumps(full_report(d["y_true"], (d["y_prob"] >= 0.5).astype(int), d["y_prob"]), indent=2))
