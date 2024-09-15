from typing import List

from oli.math.math_utility import pretty_print_matrix
from oli.ml.Functions import softmax
from oli.ml.Loss import mean_squared_error_loss_categorical
from oli.ml.deep_learning.LinearLayer import LinearLayer


class NeuralNetwork:
    layers: List[LinearLayer]

    def __init__(self, *args):
        for arg in args:
            assert isinstance(arg, LinearLayer)
        self.layers = args

    def train(self, X: List[List[float]], y: List[int], epochs: int, batch_size: int, learning_rate: float):
        """
        :param X: Data to train on.
        :param y: Labels corresponding to the data.
        :param epochs: The number of epochs to train the NN for.
        :param batch_size: Size of a training batch.
        :param learning_rate: The learning rate (amount by which the weights are adjusted).
        """
        count = 0
        for epoch in range(epochs):
            for index in range(len(X)):
                curr_x: List[float] = X[index]
                curr_y = y[index]
                # Prediction itself not used, as the activations of each layer are accessed directly
                activation: List[float] = self.predict(curr_x)
                probs: List[float] = softmax(activation)
                print(
                    f"\nLoss:{mean_squared_error_loss_categorical(probs, y[index])}\tLabel: {curr_y}\nPrediction: {probs}\nActivation: {activation}\n")
                self.backprop(learning_rate, curr_x, curr_y)
                count += 1
                if count == 1000:  # TODO: For debugging (?)
                    return

    def backprop(self, learning_rate: float, x: List[float], y: int):
        activation_cost_effect = None
        for index in reversed(range(len(self.layers))):
            print(y)
            activation_cost_effect = self.layers[index].backprop(
                previous_activation=x if index == 0 else self.layers[index - 1].activation_a,
                learning_rate=learning_rate,
                label=y,
                activation_cost_effect=activation_cost_effect
            )

    def predict(self, x: List[float]) -> List[float]:
        curr_x: List[float] = x
        log = False
        for (index, layer) in enumerate(self.layers):
            if index == 3 and log:
                pretty_print_matrix(curr_x, "X: ")
                pretty_print_matrix(layer.W, "Weights: ")
            curr_x = layer.forward(curr_x)
            if index == 0 and log:
                pretty_print_matrix(curr_x, "Activation: ")

        return curr_x

    def predict_multiple(self, X: List[List[float]]) -> List[List[float]]:
        predictions: List[List[float]] = []
        for x in X:
            predictions.append(self.predict(x))
        return predictions
