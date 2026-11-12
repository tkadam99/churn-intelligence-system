import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.prepare_data import load_prepared_data, split_data


def bootstrap_test_metrics(
    y_true,
    probabilities,
    threshold,
    n_resamples=1000,
    random_state=42,
):
    labels = np.asarray(y_true)
    scores = np.asarray(probabilities)
    rng = np.random.default_rng(random_state)
    metric_samples = {
        "roc_auc": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    # A bootstrap repeatedly samples test rows with replacement to show sampling variation.
    # These intervals reflect test-set sampling only, not variation from training the model again.
    for _ in range(n_resamples):
        indices = rng.choice(len(labels), size=len(labels), replace=True)
        sampled_labels = labels[indices]
        sampled_probabilities = scores[indices]
        sampled_predictions = (sampled_probabilities >= threshold).astype(int)

        metric_samples["roc_auc"].append(
            roc_auc_score(sampled_labels, sampled_probabilities)
        )
        metric_samples["precision"].append(
            precision_score(sampled_labels, sampled_predictions, zero_division=0)
        )
        metric_samples["recall"].append(
            recall_score(sampled_labels, sampled_predictions, zero_division=0)
        )
        metric_samples["f1"].append(
            f1_score(sampled_labels, sampled_predictions, zero_division=0)
        )

    observed_predictions = (scores >= threshold).astype(int)
    observed_values = {
        "roc_auc": roc_auc_score(labels, scores),
        "precision": precision_score(labels, observed_predictions, zero_division=0),
        "recall": recall_score(labels, observed_predictions, zero_division=0),
        "f1": f1_score(labels, observed_predictions, zero_division=0),
    }

    return {
        metric: {
            "observed": float(observed_values[metric]),
            "interval_low": float(np.percentile(values, 2.5)),
            "interval_high": float(np.percentile(values, 97.5)),
            "bootstrap_std": float(np.std(values)),
        }
        for metric, values in metric_samples.items()
    }


if __name__ == "__main__":
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
    intervals = bootstrap_test_metrics(y_test, probabilities, threshold)
    oof_estimates = config["out_of_fold_training_estimates"]

    print(f"Frozen threshold: {threshold:.2f}")
    print(f"Bootstrap resamples: 1000")
    print(
        f"{'Metric':<18} {'Observed':>10} {'95% interval low':>18} "
        f"{'95% interval high':>19} {'OOF estimate inside?':>22}"
    )
    for metric, values in intervals.items():
        oof_estimate = oof_estimates.get(metric)
        inside_interval = (
            oof_estimate is not None
            and values["interval_low"] <= oof_estimate <= values["interval_high"]
        )
        inside_display = (
            "Yes" if inside_interval else "No" if oof_estimate is not None else "-"
        )
        print(
            f"{metric:<18} {values['observed']:>10.4f} "
            f"{values['interval_low']:>18.4f} "
            f"{values['interval_high']:>19.4f} "
            f"{inside_display:>22}"
        )

    saved_results = {
        "frozen_threshold": threshold,
        "n_test_rows": len(y_test),
        "n_test_churners": int(np.asarray(y_test).sum()),
        "n_resamples": 1000,
        "random_state": 42,
        "metrics": intervals,
        "out_of_fold_training_estimates": oof_estimates,
    }
    results_path = project_root / "reports" / "final_test_confidence_intervals.json"
    results_path.write_text(json.dumps(saved_results, indent=2), encoding="utf-8")
    print(f"Saved confidence intervals: {results_path}")
