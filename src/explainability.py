import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.prepare_data import load_prepared_data, split_data


def compute_shap_values(pipeline, X_train):
    pipeline_steps = pipeline.named_steps
    print(f"Pipeline step names: {list(pipeline_steps)}")

    scaler = next(
        (step for step in pipeline_steps.values() if isinstance(step, StandardScaler)),
        None,
    )
    logistic_model = next(
        (
            step
            for step in pipeline_steps.values()
            if isinstance(step, LogisticRegression)
        ),
        None,
    )
    if scaler is None or logistic_model is None:
        raise ValueError(
            "The fitted pipeline must contain a StandardScaler and LogisticRegression."
        )

    # Transform only: reuse the fitted scaler so explanations match the frozen model.
    scaled_values = scaler.transform(X_train)
    scaled_X_train = pd.DataFrame(
        scaled_values,
        columns=X_train.columns,
        index=X_train.index,
    )
    background = shap.maskers.Independent(
        scaled_X_train,
        max_samples=len(scaled_X_train),
    )
    explainer = shap.LinearExplainer(logistic_model, background)
    shap_values = explainer.shap_values(scaled_X_train)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    # SHAP values are in log-odds: positive values push toward churn.
    # For one customer, the base value plus all feature SHAP values equals the raw model score.
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
    shap_table = pd.DataFrame(
        shap_values,
        columns=X_train.columns,
        index=X_train.index,
    )
    return shap_table, base_value


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
    X_train, _, _, _ = split_data(prepared_data)

    shap_table, base_value = compute_shap_values(pipeline, X_train)
    print(f"Base value (log-odds): {base_value:.4f}")
    shape_passed = (
        len(shap_table) == len(X_train)
        and list(shap_table.columns) == config["feature_names"]
    )
    print(
        "SHAP shape and feature-name check: "
        f"{'PASS' if shape_passed else 'FAIL'} "
        f"(rows: {len(shap_table)} / {len(X_train)}, "
        f"columns match config: {list(shap_table.columns) == config['feature_names']})"
    )
    if not shape_passed:
        raise RuntimeError("The SHAP output shape or feature names do not match.")

    raw_scores = pipeline.decision_function(X_train)
    reconstructed_scores = base_value + shap_table.sum(axis=1).to_numpy()
    largest_difference = float(
        np.max(np.abs(reconstructed_scores - raw_scores))
    )
    additivity_passed = largest_difference <= 1e-6
    print(
        "Additivity check: "
        f"{'PASS' if additivity_passed else 'FAIL'} "
        f"(largest absolute difference: {largest_difference:.12g})"
    )
    if not additivity_passed:
        raise RuntimeError("SHAP values do not add up to the model raw scores.")

    scaler = next(
        step
        for step in pipeline.named_steps.values()
        if isinstance(step, StandardScaler)
    )
    logistic_model = next(
        step
        for step in pipeline.named_steps.values()
        if isinstance(step, LogisticRegression)
    )
    scaled_X_train = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    correlations = shap_table.corrwith(scaled_X_train)
    coefficients = pd.Series(
        logistic_model.coef_[0],
        index=X_train.columns,
    )
    nonzero_features = coefficients[coefficients != 0].index
    sign_agreement = sum(
        np.isfinite(correlations[feature])
        and np.sign(correlations[feature]) == np.sign(coefficients[feature])
        for feature in nonzero_features
    )
    sign_check_passed = sign_agreement == len(nonzero_features)
    print(
        "Sign agreement check: "
        f"{'PASS' if sign_check_passed else 'FAIL'} "
        f"({sign_agreement} out of {len(nonzero_features)} "
        "nonzero-coefficient features)"
    )
    if not sign_check_passed:
        raise RuntimeError("SHAP and coefficient signs do not agree.")

    mean_absolute_shap = shap_table.abs().mean().sort_values(ascending=False)
    top_features = pd.DataFrame(
        {
            "feature": mean_absolute_shap.index,
            "mean_absolute_shap": mean_absolute_shap.values,
            "scaled_feature_coefficient": coefficients.loc[
                mean_absolute_shap.index
            ].values,
        }
    )
    print("\nTop 10 features by mean absolute SHAP value:")
    print(top_features.head(10).round(4).to_string(index=False))
