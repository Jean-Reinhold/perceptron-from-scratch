import numpy as np
from attrs import define, field


@define
class Perceptron:
    learning_rate: float = field(default=0.01)
    epochs: int = field(default=1000)

    weights: np.array = field(default=None)
    bias: np.array = field(default=None)

    loss: str = field(default="mse")
    activation: str = field(default="sigmoid")

    def fit(self, X: np.array, y: np.array) -> None:
        _, num_features = X.shape

        self.weights = np.random.random(num_features)
        self.bias = 1

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._activate(linear_model)

            dw, db = self._compute_gradients(X, y, y_pred)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.array) -> np.array:
        linear_model = (X @ self.weights) + self.bias
        y_pred = self._activate(linear_model)
        y_pred_classes = np.where(y_pred > 0.5, 1, 0)

        return y_pred_classes

    def _activate(self, x: np.array) -> np.array:
        if self.activation == "sigmoid":
            return self._sigmoid(x)

        if self.activation == "relu":
            return self._relu(x)

        raise ValueError("Invalid activation function.")

    def _compute_gradients(
        self, X: np.array, y: np.array, y_pred: np.array
    ) -> np.array:
        if self.loss == "mse":
            dw = (2 / len(y)) * (X.T @ (y_pred - y))
            db = (2 / len(y)) * np.sum(y_pred - y)
            return dw, db

        if self.loss == "log_loss":
            dw = (1 / len(y)) * (X.T @(self._sigmoid(y_pred) - y))
            db = (1 / len(y)) * np.sum(self._sigmoid(y_pred) - y)
            return dw, db

        raise ValueError("Invalid loss function.")

    def _sigmoid(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    def _relu(self, x: np.array) -> np.array:
        return np.maximum(0, x)