import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.baseline import evaluate_dummy_baseline, evaluate_logistic_regression
from src.keras_model import evaluate_keras_model
from src.prepare_data import load_prepared_data, split_data
from src.tree_models import evaluate_random_forest, evaluate_xgboost


class ChurnNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 32),  # Learns 32 combinations of the customer features.
            nn.ReLU(),  # Keeps positive values and changes negative values to zero.
            nn.Dropout(0.3),  # Randomly hides some values during training to reduce overfitting.
            nn.Linear(32, 16),  # Learns 16 combinations from the earlier layer.
            nn.ReLU(),  # Adds a simple nonlinear rule to the hidden layer.
            nn.Linear(16, 1),  # Produces one raw churn score.
        )

    def forward(self, inputs):
        # Keep a raw logit; BCEWithLogitsLoss applies sigmoid more stably during training.
        return self.layers(inputs)


def evaluate_torch_model(X_train, y_train):
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

        train_features = torch.tensor(X_fold_train_scaled, dtype=torch.float32)
        train_targets = torch.tensor(
            y_fold_train.to_numpy(), dtype=torch.float32
        ).reshape(-1, 1)
        validation_features = torch.tensor(
            X_fold_validation_scaled, dtype=torch.float32
        )
        validation_targets = torch.tensor(
            y_fold_validation.to_numpy(), dtype=torch.float32
        ).reshape(-1, 1)
        training_data = TensorDataset(train_features, train_targets)
        training_batches = DataLoader(
            training_data,
            batch_size=32,
            shuffle=True,
        )

        # This makes runs similar, but they are not guaranteed to be identical.
        torch.manual_seed(42)
        model = ChurnNet(train_features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.BCEWithLogitsLoss()
        model.train()  # Training mode enables Dropout for each training batch.

        for _ in range(30):
            for batch_features, batch_targets in training_batches:
                optimizer.zero_grad()  # Clear old gradients before calculating this batch's gradients.
                logits = model(batch_features)  # Forward pass: calculate predictions for this batch.
                loss = loss_function(logits, batch_targets)  # Measure prediction error against labels.
                loss.backward()  # Backpropagation: calculate gradients for the model parameters.
                optimizer.step()  # Update the parameters using the calculated gradients.

        model.eval()  # Evaluation mode disables Dropout for consistent predictions.
        # No gradients are needed for validation predictions, which saves memory and work.
        with torch.no_grad():
            validation_logits = model(validation_features)
            probabilities = torch.sigmoid(validation_logits).squeeze(1).numpy()

        predictions = (probabilities >= 0.5).astype(int)
        validation_labels = y_fold_validation.to_numpy()
        score_values["accuracy"].append(
            accuracy_score(validation_labels, predictions)
        )
        score_values["precision"].append(
            precision_score(validation_labels, predictions, zero_division=0)
        )
        score_values["recall"].append(
            recall_score(validation_labels, predictions, zero_division=0)
        )
        score_values["f1"].append(
            f1_score(validation_labels, predictions, zero_division=0)
        )
        score_values["roc_auc"].append(
            roc_auc_score(validation_labels, probabilities)
        )

    return {
        score_name: sum(values) / len(values)
        for score_name, values in score_values.items()
    }


if __name__ == "__main__":
    import time

    start_time = time.perf_counter()
    prepared_data = load_prepared_data()
    X_train, _, y_train, _ = split_data(prepared_data)
    model_scores = {
        "Baseline": evaluate_dummy_baseline(X_train, y_train),
        "Logistic Regression": evaluate_logistic_regression(X_train, y_train),
        "Random Forest": evaluate_random_forest(X_train, y_train),
        "XGBoost": evaluate_xgboost(X_train, y_train),
        "Keras": evaluate_keras_model(X_train, y_train),
        "PyTorch": evaluate_torch_model(X_train, y_train),
    }

    print(
        f"{'Score':<12} {'Baseline':>10} {'Logistic Regression':>20} "
        f"{'Random Forest':>15} {'XGBoost':>10} {'Keras':>10} {'PyTorch':>10}"
    )
    for score_name in model_scores["Baseline"]:
        print(
            f"{score_name:<12} "
            f"{model_scores['Baseline'][score_name]:>10.3f} "
            f"{model_scores['Logistic Regression'][score_name]:>20.3f} "
            f"{model_scores['Random Forest'][score_name]:>15.3f} "
            f"{model_scores['XGBoost'][score_name]:>10.3f} "
            f"{model_scores['Keras'][score_name]:>10.3f} "
            f"{model_scores['PyTorch'][score_name]:>10.3f}"
        )
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} seconds")
