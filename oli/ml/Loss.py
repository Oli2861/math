import numpy as np


def least_squares_element_wise(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Returns the least square loss between the prediction and label: (pred - actual)^2."""
    return (label - pred) ** 2


def least_squares(pred: np.ndarray, label: np.ndarray) -> float:
    """Returns the least square loss between the prediction and label: (pred - actual)^2."""
    return ((label - pred) ** 2).sum()


def absolute_value_loss_element_wise(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Returns the absolute value loss between the prediction and label: |pred - actual|."""
    return abs(label - pred)


def absolute_value_loss(pred: np.ndarray, label: np.ndarray) -> float:
    """Returns the absolute value loss between the prediction and label: |pred - actual|."""
    return np.sum(abs(label - pred))


def mean_squared_error_loss_categorical(pred: list[float], label: int) -> float:
    """
    Calculates the mean squared error loss for categorical values.
    Uses 0.01 as target for miss-classification and 0.99 for correct label.
    :param pred: The list of predictions.
    :param label: The label.
    :return: Squared error loss.
    """
    target = [0.01 for i in range(len(pred))]
    target[label] = 0.99
    sum = 0
    for i in range(len(target)):
        sum += (target[i] - pred[i]) ** 2
    return sum / len(target)


def categorical_cross_entropy_loss(pred: list[float], label: int) -> float:
    """
    Calculates the cross entropy loss for categorical values.
    Uses 0.01 as target for miss-classification and 0.99 for correct label.
    :param pred: The list of predictions.
    :param label: The label.
    :return: Cross entropy loss.
    """
    target = [0.01 for i in range(len(pred))]
    target[label] = 0.99
    sum = 0
    for i in range(len(target)):
        sum += target[i] * np.log(pred[i] + 1e-10)
    return -sum


def derivative_mean_squared_error_loss_categorical(pred: float, label: float) -> float:
    return 2 * (pred - label)


def derivative_categorical_cross_entropy_loss(pred: float, label: float) -> float:
    """
    Calculates the derivative of the cross entropy loss for categorical values.
    https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    :param pred: The prediction.
    :param label: The label.
    :return: The derivative of the cross entropy loss.
    """
    return pred - label
