from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from src.prepare_data import load_prepared_data, split_data


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


if __name__ == "__main__":
    prepared_data = load_prepared_data()
    X_train, _, _, _ = split_data(prepared_data)
    n_features = X_train.shape[1]
    model = build_keras_model(n_features)

    model.summary()
    print(f"Total parameters: {model.count_params()}")
