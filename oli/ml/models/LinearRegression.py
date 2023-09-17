import numpy as np

class LinearRegression:
    alpha: float
    epochs: int
    X: np.ndarray
    y: np.ndarray
    W: np.ndarray
    w_0: float

    def __init__(self, alpha: float, epochs: int):
        self.alpha = alpha
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        # Initialize weights matching the number of features.
        self.W = np.zeros(X.shape[1])
        # Initialize intercept / bias.
        self.w_0 = 0
        self.learn()

    def learn(self):
        # Adjust weights W and intercept / bias w_0 for a number of epochs by learning-rate-sized steps into the
        # direction of the gradient.
        for i in range(self.epochs):
            # Predict y_hat = h(x) = W^T * X + w_0
            y_hat: np.ndarray = self.predict(self.X)
            loss: np.ndarray = y_hat - self.y
            # Adjust weights W into a learning rate (alpha) - sized step into the direction of the gradient.
            self.W = self.W - self.alpha * loss.dot(self.X)
            # Adjust intercept / bias w_0 into a learning rate (alpha) - sized step into the direction of the gradient.
            self.w_0 = self.w_0 - self.alpha * loss.sum()

    def predict(self, X: np.ndarray):
        # h(x) = W^T * X + w_0
        return X.dot(self.W) + self.w_0
