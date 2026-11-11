import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.baseline import build_logistic_pipeline
from src.prepare_data import load_prepared_data, split_data
from src.threshold_analysis import get_oof_probabilities, threshold_table


def summarize_probabilities(name, y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    thresholds = np.arange(0.10, 0.901, 0.05)
    threshold_results = threshold_table(y_true, probabilities, thresholds)
    best_f1_row = threshold_results.loc[threshold_results["f1"].idxmax()]

    return {
        "name": name,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "flagged_customers": int(predictions.sum()),
        "best_f1": best_f1_row["f1"],
        "best_f1_threshold": best_f1_row["threshold"],
    }


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)

    default_probabilities = get_oof_probabilities(
        build_logistic_pipeline(),
        X_train,
        y_train,
    )
    balanced_probabilities = get_oof_probabilities(
        build_logistic_pipeline(class_weight="balanced"),
        X_train,
        y_train,
    )
    summaries = [
        summarize_probabilities("Default", y_train, default_probabilities),
        summarize_probabilities("Balanced", y_train, balanced_probabilities),
    ]

    print(pd.DataFrame(summaries).round(3).to_string(index=False))
