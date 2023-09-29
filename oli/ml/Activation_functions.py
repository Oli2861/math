import numpy as np


def sigmoid(x: float) -> float:
    """
    Sigmoid activation function: 1 / (1 + e^(-x))
    :param x: Number to be sigmoided.
    :return: Result of sigmoiding x.
    """
    return 1 / (1 + np.exp(-x))
