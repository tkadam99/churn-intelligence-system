import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.baseline import evaluate_dummy_baseline, evaluate_logistic_regression
from src.prepare_data import load_prepared_data, split_data
from src.tree_models import evaluate_random_forest, evaluate_xgboost


def build_keras_model(n_features):
    model = Sequential(
        [
            Input(shape=(n_features,)),  # Accepts one value for each input feature.
            Dense(32, activation="relu"),  # Learns 32 useful combinations of the inputs.
            Dropout(0.3),  # Randomly hides some values during training to reduce overfitting.
            Dense(16, activation="relu"),  # Learns 16 more combinations from the earlier layer.
            Dense(1, activation="sigmoid"),  # Produces one churn probability between 0 and 1.
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_keras_model(X_train, y_train):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score_values = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }

    for train_indices, validation_indices in folds.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_indices]
        X_fold_validation = X_train.iloc[validation_indices]
        y_fold_train = y_train.iloc[train_indices]
        y_fold_validation = y_train.iloc[validation_indices]

        # Fit only on this fold's training rows to avoid leaking validation information.
        scaler = StandardScaler()
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
        X_fold_validation_scaled = scaler.transform(X_fold_validation)

        # Reset initialization and dropout randomness; runs are similar, not guaranteed identical.
        tf.keras.utils.set_random_seed(42)
        model = build_keras_model(X_fold_train_scaled.shape[1])
        # Early stopping would need its own held-out part of this fold's training data.
        model.fit(
            X_fold_train_scaled,
            y_fold_train,
            epochs=30,
            batch_size=32,
            verbose=0,
        )

        probabilities = model.predict(X_fold_validation_scaled, verbose=0).ravel()
        predictions = (probabilities >= 0.5).astype(int)
        score_values["accuracy"].append(
            accuracy_score(y_fold_validation, predictions)
        )
        score_values["precision"].append(
            precision_score(y_fold_validation, predictions, zero_division=0)
        )
        score_values["recall"].append(
            recall_score(y_fold_validation, predictions, zero_division=0)
        )
        score_values["f1"].append(
            f1_score(y_fold_validation, predictions, zero_division=0)
        )
        score_values["roc_auc"].append(
            roc_auc_score(y_fold_validation, probabilities)
        )

    return {
        score_name: sum(values) / len(values)
        for score_name, values in score_values.items()
    }


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)
    model_scores = {
        "Baseline": evaluate_dummy_baseline(X_train, y_train),
        "Logistic Regression": evaluate_logistic_regression(X_train, y_train),
        "Random Forest": evaluate_random_forest(X_train, y_train),
        "XGBoost": evaluate_xgboost(X_train, y_train),
        "Keras": evaluate_keras_model(X_train, y_train),
    }

    print(
        f"{'Score':<12} {'Baseline':>10} {'Logistic Regression':>20} "
        f"{'Random Forest':>15} {'XGBoost':>10} {'Keras':>10}"
    )
    for score_name in model_scores["Baseline"]:
        print(
            f"{score_name:<12} "
            f"{model_scores['Baseline'][score_name]:>10.3f} "
            f"{model_scores['Logistic Regression'][score_name]:>20.3f} "
            f"{model_scores['Random Forest'][score_name]:>15.3f} "
            f"{model_scores['XGBoost'][score_name]:>10.3f} "
            f"{model_scores['Keras'][score_name]:>10.3f}"
        )
