import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.001, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize Weights and Bias

        self.weights = np.random.randn(n_features, 1)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_output = np.dot(X, self.weights).flatten() + self.bias
            y_pred = self.sigmoid(linear_output)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y).reshape(-1, 1))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= dw * self.lr
            self.bias -= db * self.lr

    def predict(self, X):
        linear_output = np.dot(X, self.weights).flatten() + self.bias
        y_pred = self.sigmoid(linear_output)
        return y_pred
