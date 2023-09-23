import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from dataclasses import dataclass
import plotly.graph_objects as go

@dataclass
class Distribution:
    """
    1-d distribution
    """

    def __init__(self, mu: float, variance: float, name: str):
        self.mu: float = mu
        self.variance: float = variance
        self.name = name
        self.sigma: float = math.sqrt(variance)
        self.pdf = np.linspace(self.mu - 3 * self.sigma, self.mu + 3 * self.sigma, 100)

    def __str__(self):
        return f"mu = {self.mu:.4f}, variance = {self.variance:.4f}, sigma = {self.sigma:.4f}, name = {self.name}"


class DisitrbutionOperations:
    @staticmethod
    def plot_distributions(*distributions: Distribution):
        """
        Plots a variable amount of 1-d distributions.
        :param distributions The distributions to be plotted.
        """
        names = []
        for d in distributions:
            plt.plot(d.pdf, stats.norm.pdf(d.pdf, d.mu, d.sigma))
            names.append(d.name)
            print(str(d))
        plt.legend(names)
        plt.show()

    @staticmethod
    def plot_distributions_plotly(*distributions: Distribution):
        """
        Plots a variable amount of 1-d distributions using Plotly.
        :param distributions The distributions to be plotted.
        """
        # Set up plot layout
        layout = go.Layout(
            xaxis=dict(title='X'),
            yaxis=dict(title='PDF'),
            legend=dict(orientation='h')
        )
        data = []

        # Loop through each distribution and add to plot data
        for d in distributions:
            # Set up PDF trace
            pdf_trace = go.Scatter(
                x=d.pdf,
                y=stats.norm.pdf(d.pdf, d.mu, d.sigma),
                mode='lines',
                name=d.name
            )

            # Set up CDF trace
            x = np.linspace(d.mu - 4 * d.sigma, d.mu + 4 * d.sigma, 1000)
            cdf_trace = go.Scatter(
                x=x,
                y=stats.norm.cdf(x, d.mu, d.sigma),
                mode='lines',
                name=d.name + ' CDF'
            )

            # Add traces to data list
            data.append(pdf_trace)
            data.append(cdf_trace)

        # Create and show figure
        fig = go.Figure(data=data, layout=layout)
        fig.show()

    @staticmethod
    def fusion(l1: Distribution, l2: Distribution) -> Distribution:
        """
        Combines two distribution using the fusion formula.
        :param l1 The first distribution.
        :param l2 The second distribution.
        :return Fused distribution.
        """
        sigma = 1 / (1 / l1.sigma + 1 / l2.sigma)
        mu = sigma * (1 / l1.sigma * l1.mu + 1 / l2.sigma * l2.mu)
        return Distribution(mu, sigma, f"Fused from {l1.name} and {l2.name}")

    @staticmethod
    def linear_transform(old: Distribution, A: np.ndarray, b: np.ndarray):
        # Transform old mu into 2x1 column vector for multiplication
        old_mu = np.full((2, 1), old.mu)
        # Make b from row to a columnn vector
        b_col = b[:, np.newaxis]

        mu_x = A @ old_mu + b_col

        # print(f"Old mean:\n{old_mu}")
        # print(f"B as column vector:\n{b_col}\n")
        print(f"Calculation of the new mean:\n{A}\t\tA\nx\t\tx\n{old_mu}\t\told mu\n+\n{b_col}\t\tb\n=\t\t=\n{mu_x}\t\ttransformed mu\n\n")

        # Create 2x2 matrix with sigma on the diagonal
        old_sigma = np.diag([old.sigma ** 2] * 2)

        sigma_x = A @ old_sigma @ A.T
        # print(f"Old sigma:\n{old_sigma}")
        print(
            f"Calculation of the new std:\n{A}\t\tA\nx\t\tx\n{old_sigma}\told sigma\nx\t\tx\n{A.T}\tA.T\n=\t\t=\n{sigma_x}\ttransformed sigma")
