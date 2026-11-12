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
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
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


def add_net_benefit(
    table,
    n_churners,
    contact_cost,
    retention_success_rate,
    customer_value,
):
    results = table.copy()
    results["true_positives"] = (
        results["recall"] * n_churners
    ).round().astype(int)
    results["false_positives"] = (
        results["flagged_customers"] - results["true_positives"]
    )
    # A true hit earns expected saved revenue minus contact cost; a false alarm costs contact cost.
    # These cost numbers are ILLUSTRATIVE ASSUMPTIONS, not real company data.
    results["net_benefit"] = (
        results["true_positives"]
        * (retention_success_rate * customer_value - contact_cost)
        - results["false_positives"] * contact_cost
    )
    return results


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)
    model = build_logistic_pipeline()
    probabilities = get_oof_probabilities(model, X_train, y_train)
    thresholds = np.arange(0.10, 0.901, 0.05)
    results = threshold_table(y_train, probabilities, thresholds)

    print(results.round(3).to_string(index=False))
    print(f"\nThreshold with highest f1: {results.loc[results['f1'].idxmax(), 'threshold']:.2f}")

    scenarios = [
        ("Scenario A", 5, 0.3, 500),
        ("Scenario B", 30, 0.3, 300),
    ]
    n_churners = int(y_train.sum())
    for name, contact_cost, retention_success_rate, customer_value in scenarios:
        scenario_results = add_net_benefit(
            results,
            n_churners,
            contact_cost,
            retention_success_rate,
            customer_value,
        )
        print(f"\n{name} (illustrative assumptions):")
        print(
            scenario_results[
                [
                    "threshold",
                    "flagged_customers",
                    "recall",
                    "precision",
                    "net_benefit",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
        best_row = scenario_results.loc[scenario_results["net_benefit"].idxmax()]
        print(
            f"Threshold with highest net_benefit: "
            f"{best_row['threshold']:.2f} ({best_row['net_benefit']:.0f})"
        )
