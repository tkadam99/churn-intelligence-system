import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.baseline import build_logistic_pipeline
from src.prepare_data import load_prepared_data, split_data


def get_oof_probabilities(model, X_train, y_train):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Out-of-fold scores keep each row's label out of the model that predicts it.
    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=folds,
        method="predict_proba",
    )
    return probabilities[:, 1]


def threshold_table(y_true, probabilities, thresholds):
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "flagged_customers": int(predictions.sum()),
                "precision": precision_score(
                    y_true, predictions, zero_division=0
                ),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["threshold", "flagged_customers", "precision", "recall", "f1"],
    )


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)
    model = build_logistic_pipeline()
    probabilities = get_oof_probabilities(model, X_train, y_train)
    thresholds = np.arange(0.10, 0.901, 0.05)
    results = threshold_table(y_train, probabilities, thresholds)

    print(results.round(3).to_string(index=False))
