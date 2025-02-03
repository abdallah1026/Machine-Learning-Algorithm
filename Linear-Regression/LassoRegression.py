import numpy as np


class LassoRegression:

    def __init__(self, lr=0.001, n_iterations=1000, alpha = 1.0):
        self.lr = lr
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize Weights and Bias

        self.weights = np.random.randn(n_features, 1)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights).flatten() + self.bias

            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - y).reshape(-1, 1)) + self.alpha * np.sign(self.weights))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= dw * self.lr
            self.bias -= db * self.lr

    def predict(self, X):
        y_pred = np.dot(X, self.weights).flatten() + self.bias

        return y_pred
