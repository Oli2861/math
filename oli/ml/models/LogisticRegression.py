import numpy as np

from oli.ml.Activation_functions import sigmoid


class LogisticRegression:
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
        # Initialize weights matching number of features
        self.W = np.zeros(X.shape[1])
        self.w_0 = 0
        self.learn()

    def learn(self):
        for epoch in range(self.epochs):
            # Predict
            y_hat: float = self.predict(self.X)
            cost: float = LogisticRegression.cost_function(self.y, y_hat)
            self.w_0: float = self.w_0 - self.alpha * (1 / len(self.y)) * np.sum(y_hat - self.y)
            self.W: np.ndarray = self.W - self.alpha * (1 / len(self.y)) * np.dot((y_hat - self.y), self.X)

            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Cost: {cost}')

    def predict(self, X: np.ndarray):
        """
        h_w(x) = sigmoid(wx^T+w_0)
        :param X: Data to make predictions for.
        :return: Predicted value based on the hypothesis the model learned.
        """
        return sigmoid(np.dot(self.W, X.T) + self.w_0)

    @staticmethod
    def cost_function(label: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculates the cost function for logistic regression.
        :param label: The true labels.
        :param prediction: The predicted labels.
        :return: The cost.
        """
        m: int = len(label)
        return -1 / m * np.sum(label * np.log(prediction) + (1 - label) * np.log(1 - prediction))