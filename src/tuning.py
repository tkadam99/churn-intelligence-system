import time

import pandas as pd
from xgboost import XGBClassifier
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


def tune_xgboost(X_train, y_train):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # One worker per fit avoids competing with GridSearchCV's parallel workers.
    model = XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    search = GridSearchCV(
        estimator=model,
        param_grid={
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [200, 300],
        },
        scoring="roc_auc",
        cv=folds,
        refit=False,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    return pd.DataFrame(
        {
            "max_depth": search.cv_results_["param_max_depth"].astype(int),
            "learning_rate": search.cv_results_[
                "param_learning_rate"
            ].astype(float),
            "n_estimators": search.cv_results_[
                "param_n_estimators"
            ].astype(int),
            "mean_roc_auc": search.cv_results_["mean_test_score"],
            "std_roc_auc": search.cv_results_["std_test_score"],
            "rank": search.cv_results_["rank_test_score"],
        }
    )


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)
    logistic_results = tune_logistic_regression(X_train, y_train)

    print("Logistic regression tuning:")
    print(logistic_results.sort_values("C").round(4).to_string(index=False))

    start_time = time.perf_counter()
    xgboost_results = tune_xgboost(X_train, y_train)
    elapsed_time = time.perf_counter() - start_time

    ranked_results = xgboost_results.sort_values("rank")
    print("\nXGBoost tuning:")
    print(ranked_results.round(4).to_string(index=False))

    step_30_row = xgboost_results.loc[
        (xgboost_results["max_depth"] == 3)
        & (xgboost_results["learning_rate"] == 0.05)
        & (xgboost_results["n_estimators"] == 300)
    ].iloc[0]
    best_xgboost_mean = xgboost_results["mean_roc_auc"].max()
    best_logistic_mean = logistic_results["mean_roc_auc"].max()

    print(
        "\nStep 30 setting: "
        f"max_depth={step_30_row['max_depth']}, "
        f"learning_rate={step_30_row['learning_rate']:.2f}, "
        f"n_estimators={step_30_row['n_estimators']}, "
        f"mean_roc_auc={step_30_row['mean_roc_auc']:.4f}, "
        f"rank={int(step_30_row['rank'])}"
    )
    print(
        "Gap between best and Step 30 mean ROC-AUC: "
        f"{best_xgboost_mean - step_30_row['mean_roc_auc']:.4f}"
    )
    print(
        "Best XGBoost mean ROC-AUC: "
        f"{best_xgboost_mean:.4f}; best logistic regression mean ROC-AUC: "
        f"{best_logistic_mean:.4f}; difference (XGBoost - logistic): "
        f"{best_xgboost_mean - best_logistic_mean:.4f}"
    )
    print(f"XGBoost search elapsed time: {elapsed_time:.2f} seconds")
