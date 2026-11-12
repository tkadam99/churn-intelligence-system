import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.baseline import build_logistic_pipeline
from src.prepare_data import load_prepared_data, split_data


def tune_logistic_regression(X_train, y_train):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=build_logistic_pipeline(),
        param_grid={"logistic_regression__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
        scoring="roc_auc",
        cv=folds,
        refit=False,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    return pd.DataFrame(
        {
            "C": search.cv_results_["param_logistic_regression__C"].astype(float),
            "mean_roc_auc": search.cv_results_["mean_test_score"],
            "std_roc_auc": search.cv_results_["std_test_score"],
            "rank": search.cv_results_["rank_test_score"],
        }
    )


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)
    results = tune_logistic_regression(X_train, y_train)

    print(results.sort_values("C").round(4).to_string(index=False))
