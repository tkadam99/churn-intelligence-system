from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.prepare_data import load_prepared_data, split_data


def evaluate_dummy_baseline(X_train, y_train):
    model = DummyClassifier(strategy="most_frequent")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=folds,
        scoring=scoring,
    )
    return {
        score_name: scores[f"test_{score_name}"].mean()
        for score_name in scoring
    }


def build_logistic_pipeline(class_weight=None):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def evaluate_logistic_regression(X_train, y_train):
    model = build_logistic_pipeline()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=folds,
        scoring=scoring,
    )
    return {
        score_name: scores[f"test_{score_name}"].mean()
        for score_name in scoring
    }


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)
    baseline_scores = evaluate_dummy_baseline(X_train, y_train)
    logistic_scores = evaluate_logistic_regression(X_train, y_train)

    print(f"{'Score':<12} {'Baseline':>10} {'Logistic Regression':>20}")
    for score_name in baseline_scores:
        print(
            f"{score_name:<12} {baseline_scores[score_name]:>10.3f} "
            f"{logistic_scores[score_name]:>20.3f}"
        )
