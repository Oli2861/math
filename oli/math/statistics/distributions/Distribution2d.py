from typing import List

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from scipy.stats import multivariate_normal


@dataclass
class Distribution2D:
    """
    2-d distribution
    """

    def __init__(self, mu: List[float], covariance_matrix: List[float] | np.ndarray, name: str, pos: np.ndarray):
        self.mu = mu
        self.cov_matrix = covariance_matrix
        self.name = name
        self.pdf = multivariate_normal(mu, covariance_matrix).pdf(pos)

    def __str__(self):
        return f"mu = {self.mu}, variance = {self.cov_matrix}, name = {self.name}"


class Distribution2DOperations:
    @staticmethod
    def plot_distribution2d(x, y, *distributions: Distribution2D):
        """
        Plots a variable amount of 2-d distributions.
        :param distributions The distributions to be plotted.
        """
        colors = ['red', 'blue', 'green', 'purple', 'yellow']
        names = []
        fig, ax = plt.subplots(figsize=(10, 10))
        for d in distributions:
            color = colors[0]
            colors.remove(color)
            ax.contour(x, y, d.pdf, colors=[color])
            names.append(d.name)
            print(f"Plotting {str(d)} with color {color}")
        plt.show()

    @staticmethod
    def fusion_2d(l1: Distribution2D, l2: Distribution2D, pos) -> Distribution2D:
        # new cov
        cov_matrix = np.linalg.inv(np.linalg.inv(l1.cov_matrix) + np.linalg.inv(l2.cov_matrix))
        mu = cov_matrix @ (np.linalg.inv(l1.cov_matrix) @ l1.mu + np.linalg.inv(l2.cov_matrix) @ l2.mu)
        return Distribution2D(mu=mu, covariance_matrix=cov_matrix, name=f"Fused PDF from {l1.name} and {l2.name}", pos=pos)
