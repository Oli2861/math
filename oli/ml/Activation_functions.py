import numpy as np
import math


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
