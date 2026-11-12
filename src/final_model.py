import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score

from src.baseline import build_logistic_pipeline
from src.prepare_data import load_prepared_data, split_data
from src.threshold_analysis import get_oof_probabilities, threshold_table


def freeze_final_model(X_train, y_train):
    pipeline = build_logistic_pipeline()
    # C=1.0 was kept because the best tuned C beat it by only 0.0002, far less than the fold-to-fold spread.
    pipeline.set_params(logistic_regression__C=1.0)

    probabilities = get_oof_probabilities(pipeline, X_train, y_train)
    thresholds = np.arange(0.10, 0.901, 0.05)
    results = threshold_table(y_train, probabilities, thresholds)
    best_threshold_row = results.loc[results["f1"].idxmax()]
    threshold = round(float(best_threshold_row["threshold"]), 2)
    # threshold = the F1-best cutoff on out-of-fold TRAINING predictions, decided before the test set was used
    predictions = (probabilities >= threshold).astype(int)
    oof_estimates = {
        "precision": float(
            precision_score(y_train, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_train, predictions, zero_division=0)),
        "f1": float(f1_score(y_train, predictions, zero_division=0)),
    }

    pipeline.fit(X_train, y_train)
    config = {
        "model_name": "LogisticRegression",
        "C": 1.0,
        "scaler": "StandardScaler",
        "random_state": 42,
        "threshold": threshold,
        "threshold_rule": (
            "threshold = the F1-best cutoff on out-of-fold TRAINING "
            "predictions, decided before the test set was used"
        ),
        "n_train_rows": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
        "scikit_learn_version": sklearn.__version__,
        "out_of_fold_training_estimates": oof_estimates,
    }
    return pipeline, config


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)
    pipeline, config = freeze_final_model(X_train, y_train)

    models_directory = Path(__file__).resolve().parents[1] / "models"
    models_directory.mkdir(exist_ok=True)
    model_path = models_directory / "final_logistic_pipeline.joblib"
    config_path = models_directory / "final_config.json"

    joblib.dump(pipeline, model_path)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    printed_config = config.copy()
    printed_config["feature_names"] = f"{len(config['feature_names'])} feature names"
    print(json.dumps(printed_config, indent=2))
    print(f"Model file: {model_path} ({model_path.stat().st_size} bytes)")
    print(f"Config file: {config_path} ({config_path.stat().st_size} bytes)")
