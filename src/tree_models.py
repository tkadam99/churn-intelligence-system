from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.baseline import evaluate_dummy_baseline, evaluate_logistic_regression
from src.prepare_data import load_prepared_data, split_data


def evaluate_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
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
    model_scores = {
        "Baseline": evaluate_dummy_baseline(X_train, y_train),
        "Logistic Regression": evaluate_logistic_regression(X_train, y_train),
        "Random Forest": evaluate_random_forest(X_train, y_train),
    }

    print(
        f"{'Score':<12} {'Baseline':>10} "
        f"{'Logistic Regression':>20} {'Random Forest':>15}"
    )
    for score_name in model_scores["Baseline"]:
        print(
            f"{score_name:<12} "
            f"{model_scores['Baseline'][score_name]:>10.3f} "
            f"{model_scores['Logistic Regression'][score_name]:>20.3f} "
            f"{model_scores['Random Forest'][score_name]:>15.3f}"
        )
