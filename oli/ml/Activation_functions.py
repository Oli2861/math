import numpy as np

def sigmoid(x: float) -> float:
    """
    Sigmoid activation function: 1 / (1 + e^(-x))
    :param x: Number to be sigmoided.
    :return: Result of sigmoiding x.
    """
    return 1 / (1 + np.exp(-x))


def relu(x: float) -> float:
    return 0 if x <= 0 else x


def relu_derivative(x: float) -> float:
    return 1 if x > 0 else 0

def softmax(x: list[float]) -> list[float]:
    """
    Converts vector into probability distribution of outcomes.
    :param x: Vector.
    :return: Probability distribution of outcomes.
    """
    sum = 0.0000000000000000000000001
    for item in x:
        sum += item
    return [curr / sum for curr in x]
