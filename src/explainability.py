import json
from pathlib import Path

import joblib
import matplotlib

# Select a file-rendering backend before plotting imports so no window opens.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


def plot_shap_importance_bar(shap_df, path, top_n=15):
    mean_absolute_shap = shap_df.abs().mean().sort_values().tail(top_n)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(mean_absolute_shap.index, mean_absolute_shap.values)
    axis.set_xlabel("Mean absolute SHAP value (log-odds)")
    axis.set_ylabel("Feature")
    axis.set_title(f"Top {len(mean_absolute_shap)} Features by Mean Absolute SHAP")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_shap_beeswarm(shap_df, X_train, path, top_n=15):
    shap.summary_plot(
        shap_df.values,
        features=X_train,
        feature_names=list(shap_df.columns),
        max_display=top_n,
        show=False,
    )
    plt.gcf().savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def find_duplicate_columns(X):
    duplicate_groups = []
    checked_columns = set()

    for column in X.columns:
        if column in checked_columns:
            continue
        group = [column]
        for other_column in X.columns:
            if other_column == column or other_column in checked_columns:
                continue
            if X[column].equals(X[other_column]):
                group.append(other_column)
                checked_columns.add(other_column)
        if len(group) > 1:
            duplicate_groups.append(group)
        checked_columns.add(column)

    return duplicate_groups


def find_correlated_pairs(X, min_abs=0.7):
    correlations = X.corr(numeric_only=True)
    pairs = []

    for left_index, left_column in enumerate(correlations.columns):
        for right_column in correlations.columns[left_index + 1 :]:
            if X[left_column].equals(X[right_column]):
                continue
            correlation = float(correlations.loc[left_column, right_column])
            if abs(correlation) >= min_abs:
                pairs.append((left_column, right_column, correlation))

    return sorted(pairs, key=lambda pair: abs(pair[2]), reverse=True)


def combine_identical_columns(shap_df, groups, names):
    if len(groups) != len(names):
        raise ValueError("Each identical-column group must have one output name.")

    combined = shap_df.copy()
    insertions = []
    for group, name in zip(groups, names):
        missing_columns = [column for column in group if column not in combined]
        if missing_columns:
            raise ValueError(f"SHAP columns are missing: {missing_columns}")
        insertion_index = min(combined.columns.get_loc(column) for column in group)
        summed_values = combined[group].sum(axis=1)
        combined = combined.drop(columns=group)
        insertions.append((insertion_index, name, summed_values))

    for insertion_index, name, values in sorted(insertions, reverse=True):
        combined.insert(min(insertion_index, len(combined.columns)), name, values)

    return combined


def explain_customer(
    row_features,
    row_shap,
    base_value,
    raw_score,
    probability,
    threshold,
    top_n=5,
):
    contributions_sum = float(row_shap.sum())
    status = (
        "at or above"
        if probability >= threshold
        else "below"
    )
    lines = [
        f"Predicted churn probability: {probability:.4f}",
        f"This is {status} the frozen threshold of {threshold:.2f}.",
        f"Base value (log-odds): {base_value:.4f}",
        f"Sum of feature contributions (log-odds): {contributions_sum:.4f}",
        (
            f"Raw model score (log-odds): {raw_score:.4f} "
            f"(base + contributions = {base_value + contributions_sum:.4f})"
        ),
        "",
        f"Top {top_n} contributions pushing toward churn:",
    ]

    positive = row_shap[row_shap > 0].nlargest(top_n)
    if positive.empty:
        lines.append("None")
    for feature, contribution in positive.items():
        if feature not in row_features:
            raise ValueError(f"Customer feature value is missing for {feature!r}.")
        # Odds are p / (1 - p); a zero contribution means no change in odds.
        odds_multiplier = float(np.exp(contribution))
        lines.append(
            f"- {feature}: value={row_features[feature]}, "
            f"contribution={contribution:+.4f} log-odds; "
            f"multiplies the odds of churn by {odds_multiplier:.1f}"
        )

    lines.extend(["", f"Top {top_n} contributions pushing away from churn:"])
    negative = row_shap[row_shap < 0].nsmallest(top_n)
    if negative.empty:
        lines.append("None")
    for feature, contribution in negative.items():
        if feature not in row_features:
            raise ValueError(f"Customer feature value is missing for {feature!r}.")
        odds_multiplier = float(np.exp(contribution))
        lines.append(
            f"- {feature}: value={row_features[feature]}, "
            f"contribution={contribution:+.4f} log-odds; "
            f"multiplies the odds of churn by {odds_multiplier:.1f}"
        )

    return "\n".join(lines)


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

    figures_directory = project_root / "reports" / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)
    bar_path = figures_directory / "shap_importance_bar.png"
    beeswarm_path = figures_directory / "shap_beeswarm.png"
    plot_shap_importance_bar(shap_table, bar_path)
    plot_shap_beeswarm(shap_table, X_train, beeswarm_path)
    print(f"\nSHAP importance bar: {bar_path} ({bar_path.stat().st_size} bytes)")
    print(f"SHAP beeswarm: {beeswarm_path} ({beeswarm_path.stat().st_size} bytes)")

    print("\nGroups of exactly identical training feature columns:")
    duplicate_groups = find_duplicate_columns(X_train)
    if duplicate_groups:
        for group in duplicate_groups:
            print(group)
    else:
        print("None")

    print("\nCorrelated training feature pairs (absolute correlation >= 0.7):")
    correlated_pairs = find_correlated_pairs(X_train, min_abs=0.7)
    if correlated_pairs:
        for left_column, right_column, correlation in correlated_pairs:
            print(
                f"{left_column} | {right_column}: {correlation:.3f}"
            )
    else:
        print("None")

    identical_groups = find_duplicate_columns(X_train)
    combined_names = [
        "No internet service (7 identical columns combined)"
        if len(group) == 7
        else f"{group[0]} ({len(group)} identical columns combined)"
        for group in identical_groups
    ]
    customer_shap = combine_identical_columns(
        shap_table,
        identical_groups,
        combined_names,
    )
    probabilities = pipeline.predict_proba(X_train)[:, 1]
    raw_scores = pipeline.decision_function(X_train)
    selected_customers = [
        ("Highest risk", int(np.argmax(probabilities))),
        (
            "Near the threshold",
            int(np.argmin(np.abs(probabilities - config["threshold"]))),
        ),
        ("Lowest risk", int(np.argmin(probabilities))),
    ]

    print("\nIndividual training-customer explanations:")
    for label, position in selected_customers:
        customer_index = X_train.index[position]
        row_features = X_train.iloc[position].copy()
        row_shap = customer_shap.iloc[position]
        for group, combined_name in zip(identical_groups, combined_names):
            row_features[combined_name] = X_train.iloc[position][group[0]]

        raw_score = float(raw_scores[position])
        reconstructed_score = base_value + float(row_shap.sum())
        check_passed = abs(reconstructed_score - raw_score) <= 1e-6
        print(f"\n{label} (X_train row index: {customer_index})")
        print(
            "Customer additivity check: "
            f"{'PASS' if check_passed else 'FAIL'} "
            f"(base + contributions = {reconstructed_score:.12f}, "
            f"raw score = {raw_score:.12f})"
        )
        if not check_passed:
            raise RuntimeError(
                f"SHAP contributions do not reconstruct customer {customer_index}."
            )
        print(
            explain_customer(
                row_features,
                row_shap,
                base_value,
                raw_score,
                float(probabilities[position]),
                config["threshold"],
            )
        )
