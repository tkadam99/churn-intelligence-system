from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

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


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)
    baseline_scores = evaluate_dummy_baseline(X_train, y_train)

    for score_name, score in baseline_scores.items():
        print(f"{score_name}: {score:.3f}")
