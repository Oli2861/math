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
