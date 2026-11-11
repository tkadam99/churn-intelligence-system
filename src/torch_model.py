import torch
from torch import nn

from src.prepare_data import load_prepared_data, split_data


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


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, _, _, _ = split_data(prepared_data)
    model = ChurnNet(X_train.shape[1])

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(model)
    print(f"Total parameters: {total_parameters}")
    print(f"Trainable parameters: {trainable_parameters}")
