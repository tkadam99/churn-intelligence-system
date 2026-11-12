# This evaluation is run once. Results are reported as they come out; the model, threshold and features must not be changed in response to them.
import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.prepare_data import load_prepared_data, split_data


def evaluate_on_test(pipeline, config, X_test, y_test):
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold = config["threshold"]
    predictions = (probabilities >= threshold).astype(int)
    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    )

    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "flagged_customers": int(predictions.sum()),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
    }


def reference_at_default_threshold(pipeline, X_test, y_test):
    # Reference only; the frozen threshold stays 0.30 and is the reported result.
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "final_logistic_pipeline.joblib"
    config_path = project_root / "models" / "final_config.json"
    pipeline = joblib.load(model_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    prepared_data = load_prepared_data()
    X_train, X_test, y_train, y_test = split_data(prepared_data)

    row_count_passed = len(X_train) == config["n_train_rows"]
    column_order_passed = list(X_test.columns) == config["feature_names"]
    churn_rate_difference = abs(y_test.mean() - y_train.mean()) * 100
    churn_rate_passed = churn_rate_difference <= 0.5

    print(
        "Safety check - training row count matches config: "
        f"{'PASS' if row_count_passed else 'FAIL'}"
    )
    print(
        "Safety check - test columns match config order: "
        f"{'PASS' if column_order_passed else 'FAIL'}"
    )
    print(
        "Safety check - test churn rate within 0.5 percentage points "
        f"of training: {'PASS' if churn_rate_passed else 'FAIL'} "
        f"(difference: {churn_rate_difference:.3f} percentage points)"
    )
    if not all((row_count_passed, column_order_passed, churn_rate_passed)):
        raise RuntimeError("A final-evaluation safety check failed.")

    results = evaluate_on_test(pipeline, config, X_test, y_test)
    oof_estimates = config["out_of_fold_training_estimates"]
    comparison_rows = [
        ("roc_auc", results["roc_auc"], "-"),
        ("average_precision", results["average_precision"], "-"),
        ("accuracy", results["accuracy"], "-"),
        ("precision", results["precision"], oof_estimates["precision"]),
        ("recall", results["recall"], oof_estimates["recall"]),
        ("f1", results["f1"], oof_estimates["f1"]),
        ("flagged_customers", results["flagged_customers"], "-"),
    ]

    print("\nTest results at frozen threshold:")
    print(f"Frozen threshold: {config['threshold']:.2f}")
    print(f"{'Metric':<22} {'Test value':>12} {'OOF training estimate':>24}")
    for metric, test_value, oof_value in comparison_rows:
        formatted_test = (
            f"{test_value:.4f}" if isinstance(test_value, float) else str(test_value)
        )
        formatted_oof = f"{oof_value:.4f}" if isinstance(oof_value, float) else oof_value
        print(f"{metric:<22} {formatted_test:>12} {formatted_oof:>24}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("                    Predicted no churn  Predicted churn")
    print(
        f"Actual no churn     {results['true_negatives']:>18} "
        f"{results['false_positives']:>15}"
    )
    print(
        f"Actual churn        {results['false_negatives']:>18} "
        f"{results['true_positives']:>15}"
    )
    print(f"Number of test rows: {len(X_test)}")
    print(f"Actual churners in test set: {int(y_test.sum())}")

    reference = reference_at_default_threshold(pipeline, X_test, y_test)
    print(
        "Reference only at threshold 0.50 (frozen threshold remains "
        f"{config['threshold']:.2f}): "
        f"precision={reference['precision']:.4f}, "
        f"recall={reference['recall']:.4f}, f1={reference['f1']:.4f}"
    )

    results_path = project_root / "reports" / "final_test_results.json"
    results_path.parent.mkdir(exist_ok=True)
    saved_results = {
        **results,
        "frozen_threshold": config["threshold"],
        "n_test_rows": len(X_test),
        "n_test_churners": int(y_test.sum()),
        "reference_at_threshold_0_5": reference,
    }
    results_path.write_text(json.dumps(saved_results, indent=2), encoding="utf-8")
    print(f"Saved test results: {results_path}")
