import matplotlib

# Use a non-interactive backend so plots save without opening a desktop window.
matplotlib.use("Agg")

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.prepare_data import load_prepared_data, split_data


def plot_roc_curve(y_true, probabilities, threshold, path):
    false_positive_rates, true_positive_rates, _ = roc_curve(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    )
    operating_point = (
        float(false_positives / (false_positives + true_negatives)),
        float(true_positives / (true_positives + false_negatives)),
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        false_positive_rates,
        true_positive_rates,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], "--", label="chance")
    plt.scatter(
        *operating_point,
        marker="o",
        s=70,
        label=f"threshold = {threshold:.2f}",
        zorder=3,
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve - Frozen Logistic Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return operating_point


def plot_precision_recall_curve(y_true, probabilities, threshold, path):
    precisions, recalls, _ = precision_recall_curve(y_true, probabilities)
    average_precision = average_precision_score(y_true, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    operating_point = (
        float(recall_score(y_true, predictions)),
        float(precision_score(y_true, predictions, zero_division=0)),
    )
    churn_rate = float(np.mean(y_true))

    plt.figure(figsize=(7, 6))
    plt.plot(
        recalls,
        precisions,
        label=f"Precision-recall curve (AP = {average_precision:.3f})",
    )
    plt.axhline(
        churn_rate,
        linestyle="--",
        label=f"no-skill (churn rate = {churn_rate:.3f})",
    )
    plt.scatter(
        *operating_point,
        marker="o",
        s=70,
        label=f"threshold = {threshold:.2f}",
        zorder=3,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Frozen Logistic Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return operating_point


if __name__ == "__main__":
    # These plots describe results already computed; they are not used to choose or change anything.
    project_root = Path(__file__).resolve().parents[1]
    pipeline = joblib.load(
        project_root / "models" / "final_logistic_pipeline.joblib"
    )
    config = json.loads(
        (project_root / "models" / "final_config.json").read_text(
            encoding="utf-8"
        )
    )
    prepared_data = load_prepared_data()
    _, X_test, _, y_test = split_data(prepared_data)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold = config["threshold"]

    figures_directory = project_root / "reports" / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)
    roc_path = figures_directory / "roc_curve.png"
    pr_path = figures_directory / "precision_recall_curve.png"

    roc_point = plot_roc_curve(y_test, probabilities, threshold, roc_path)
    pr_point = plot_precision_recall_curve(
        y_test, probabilities, threshold, pr_path
    )

    print(f"Frozen threshold: {threshold:.2f}")
    print(f"ROC plot: {roc_path} ({roc_path.stat().st_size} bytes)")
    print(
        f"ROC operating point (FPR, TPR): "
        f"({roc_point[0]:.4f}, {roc_point[1]:.4f})"
    )
    print(f"Precision-recall plot: {pr_path} ({pr_path.stat().st_size} bytes)")
    print(
        f"Precision-recall operating point (recall, precision): "
        f"({pr_point[0]:.4f}, {pr_point[1]:.4f})"
    )
