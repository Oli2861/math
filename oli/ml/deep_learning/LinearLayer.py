import random
import unittest
from typing import Callable, List

from torch.nn.functional import mse_loss

from oli.math.linear_algebra.matrix_operations import multiplication, transpose
from oli.math.linear_algebra.vector_operations import matrix_vector_multiplication
from oli.math.math_utility import pretty_print_matrix
from oli.ml.Activation_functions import relu_derivative, relu
from oli.ml.Loss import derivative_mean_squared_error_loss_categorical, mean_squared_error_loss_categorical
from oli.ml.deep_learning.Layer import Layer


class LinearLayer(Layer):
    activation_function: Callable[[float], float]
    derivative_activation_function: Callable[[float], float]
    derivative_cost_function: Callable[[float, float], float]
    bias: float = 1
    W = List[List[float]]
    neurons: int
    inputs: int
    test_mode: bool

    # For backpropagation
    activation_a: List[float]
    matrix_multiplication_result_z: List[float]

    def __init__(
            self,
            neurons: int,
            inputs: int,
            activation_function: Callable[[float], float],
            derivative_activation_function: Callable[[float], float],
            derivative_cost_function: Callable[[float, float], float],
            test_mode: bool = False
    ):
        """
        Create a linear layer with a fixed number of neurons for a fixed number of inputs.
        :param neurons: The number of neurons the linear layer shall contain.
        :param inputs: The number of inputs.
        :param activation_function: The activation function of the linear layer.
        """
        self.neurons = neurons
        self.inputs = inputs
        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function
        self.derivative_cost_function = derivative_cost_function
        self.test_mode = test_mode

        if test_mode:
            self.W = [[0.5 for m in range(neurons)] for n in range(inputs + 1)]
        else:
            self.W = [[random.random() * 2 - 1 for m in range(neurons)] for n in range((inputs + 1))]

        assert len(self.W) == inputs + 1
        assert len(self.W[0]) == neurons

    def forward(self, x: List[float]) -> List[List[float]]:
        """
        Forward pass through the linear layer by multiplying the weights (including bias) with the data (padded by a additional unit for the bias).
        :param x: Data used to make a prediction.
        :return: Prediction of the linear layer.
        """
        # Add bias to input to allow the forward pass to be treated as a matrix multiplication.
        assert len(x) == self.inputs
        x.insert(0, self.bias)

        if self.test_mode: print(
            f"Input count: {len(x):^6}\tWeight dimensions count: {len(self.W):^6} x {len(self.W[0]):^6} = {len(self.W) * len(self.W[0])}")

        product = [0 for i in range(self.neurons)]
        w_T = transpose(self.W)
        for neuron_index in range(self.neurons):
            sum = 0
            for i in range(self.inputs + 1):
                mult = w_T[neuron_index][i] * x[i]
                sum += mult
            product[neuron_index] = sum

        if self.test_mode:
            print("X (padded by one for bias):")
            pretty_print_matrix(x)
            print("Weights:")
            pretty_print_matrix(self.W)
            print("Weights (transposed):")
            pretty_print_matrix(transpose(self.W))
            print("Result:")
            pretty_print_matrix(product)

        assert len(product) == self.neurons, f"{len(self.W)=}, {len(product)=}"
        # Save result of the multiplication (z) for backpropagation
        self.matrix_multiplication_result_z = product

        # Apply activation function
        activation: List[float] = [self.activation_function(curr) for curr in product]
        # Save activation (a) for backpropagation
        self.activation_a = activation

        return activation

    def backprop(
            self,
            previous_activation: List[float],
            learning_rate: float,
            label: int | None = None,
            activation_cost_effect: List[float] | None = None,
            print_info: bool = False
    ) -> List[float]:
        """
        Backpropagate the error through the neural network. Adjust the weights based on the learning rate and the error received at the corresponding linear layer.
        :param previous_activation: Activation received from the previous layer / input: a^{(L-1)}
        :param label:
        :param learning_rate: Describes how large the gradient steps are.
        :param activation_cost_effect: Costs induced by the activation of this layer. If this layer is the last layer set to None in order to calculate the effect based on the loss.
        :return: Activation cost effect of the upstream layer.
        """
        activation_cost_effect_for_upstream_layer = [0 for index in range(len(previous_activation))]

        for neuron_index_j in range(self.neurons):
            for previous_activation_index_k in range(len(previous_activation)):

                # Calculate chain rule components
                # Effect of a weight change on the received matrix multiplication product
                w_on_z_effect: float = self.effect_of_weights_on_matrix_multiplication_product(
                    previous_activation[neuron_index_j])

                # Effect of matrix multiplication product change on activation
                z_on_a_effect: float = self.effect_of_matrix_multiplication_product_on_activation(
                    self.matrix_multiplication_result_z[neuron_index_j])

                prev_a_on_z_effect: float = self.effect_of_previous_activation_on_matrix_product(
                    neuron_index_j,
                    previous_activation_index_k
                )

                if neuron_index_j == 0 and print_info:
                    print("prev_a_on_z_effect", prev_a_on_z_effect)
                    print("z_on_a_effect", z_on_a_effect)

                # Effect of activation change on costs
                if activation_cost_effect is None and label is not None:
                    a_on_c0_effect: float = self.effect_of_activation_on_cost(self.activation_a[neuron_index_j], label)
                    if neuron_index_j == 0 and print_info:
                        print("a_on_c0_effect (based on label)", a_on_c0_effect)
                elif activation_cost_effect is not None:
                    a_on_c0_effect: float = activation_cost_effect[neuron_index_j]
                    if neuron_index_j == 0 and print_info:
                        print("a_on_c0_effect (based on previous activation)", a_on_c0_effect)
                else:
                    raise Exception("Illegal state.")

                # Effect of previous activation
                activation_cost_effect_for_upstream_layer[
                    previous_activation_index_k] += prev_a_on_z_effect * z_on_a_effect * a_on_c0_effect

                # Chain rule: Effect of changes of the weights on the costs
                cost_sensitivity_with_respect_to_weight_changes = w_on_z_effect * z_on_a_effect * a_on_c0_effect

                # Adjust weights
                self.W[previous_activation_index_k][neuron_index_j] = self.W[previous_activation_index_k][
                                                                          neuron_index_j] - learning_rate * cost_sensitivity_with_respect_to_weight_changes

        return activation_cost_effect_for_upstream_layer

    def effect_of_weights_on_matrix_multiplication_product(self, previous_activation: float) -> float:
        """
        The effect of the weights is given by the previous activation a^{(L-1)}.
        """
        return previous_activation

    def effect_of_matrix_multiplication_product_on_activation(self, matrix_multiplication_result_z_j: float) -> float:
        """
        The effect of the matrix multiplication product on the activation is the derivative of the activation function applied to the matrix multiplication product: \sigma ' (z^{(L)}).
        """
        return self.derivative_activation_function(matrix_multiplication_result_z_j)

    def effect_of_activation_on_cost(self, activation: float, label: float):
        """
        The effect of the activation on the cost is the derivative of the loss function. e.g. 2(a^{(L)} -y) for squared error loss
        """
        return self.derivative_cost_function(activation, label)

    def effect_of_previous_activation_on_matrix_product(self, neuron_index_j: int,
                                                        previous_activation_index_k: int) -> float:
        """
        Calculate the effect of the previous activation on the matrix multiplication product.
        It is the activation at the index of the relevant weight: Row index in the weight matrix denotes the feature index, column index denotes the neurons of the layer.
        """
        return self.W[previous_activation_index_k][neuron_index_j]


import numpy as np
import torch
import torch.nn as nn


class TestLinearLayer(unittest.TestCase):

    def test_forward(self):
        layer = LinearLayer(
            neurons=4,
            inputs=3,
            activation_function=relu,
            derivative_activation_function=relu_derivative,
            derivative_cost_function=derivative_mean_squared_error_loss_categorical,
            test_mode=True
        )
        result = layer.forward([1, 2, 3])
        expected = [3.5, 3.5, 3.5, 3.5]
        self.assertListEqual(expected, result)

    def test_forward_minimal(self):
        x = [1, 2]

        with torch.no_grad():
            pt_layer = nn.Linear(in_features=2, out_features=4)
            pt_layer.weight.copy_(torch.tensor(
                [
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.5],
                ]
            ))
            pt_layer.bias.copy_(torch.tensor([0.5, 0.5, 0.5, 0.5]))
            pt_output = pt_layer(torch.tensor(x, dtype=torch.float32))
            print(f"{pt_layer.weight=}")
            print(f"{pt_layer.bias=}")
            print(f"{pt_output=}")
            result: List[List[float]] = pt_output.tolist()
            expected = [2, 2, 2, 2]
            self.assertListEqual(expected, result)

            layer = LinearLayer(
                neurons=4,
                inputs=2,
                activation_function=relu,
                derivative_activation_function=relu_derivative,
                derivative_cost_function=derivative_mean_squared_error_loss_categorical,
                test_mode=True
            )
            result = layer.forward(x)
            self.assertListEqual(expected, result)

    def test_forward_larger(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

        with torch.no_grad():
            pt_layer = nn.Linear(in_features=len(x), out_features=6)
            pt_layer.weight.copy_(torch.tensor(
                [
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                ]
            ))
            pt_layer.bias.copy_(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
            pt_output = pt_layer(torch.tensor(x, dtype=torch.float32))
            print(f"{pt_layer.weight=}")
            print(f"{pt_layer.bias=}")
            print(f"{pt_output=}")
            pt_result: List[List[float]] = pt_output.tolist()

            layer = LinearLayer(
                neurons=6,
                inputs=len(x),
                activation_function=relu,
                derivative_activation_function=relu_derivative,
                derivative_cost_function=derivative_mean_squared_error_loss_categorical,
                test_mode=True
            )
            result = layer.forward(x)
            self.assertListEqual(pt_result, result)

    def test_bias_adding_is_bias_padding_theory(self):
        # Input layer
        x = np.array([1, 2, 3])
        # Weight matrix
        W = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            ]
        )
        # Bias vector
        b = np.array([0.1, 0.2, 0.5, 0.2, 1])
        # Compute weighted sum
        z_add = np.dot(x, W.T) + b

        # Input layer
        x = np.array([1, 2, 3])
        # Weight matrix
        W = np.array(
            [
                [0.1, 0.5, 0.5, 0.5],
                [0.2, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.2, 0.5, 0.5, 0.5],
                [1, 0.5, 0.5, 0.5],
            ]
        )
        # Pad input with ones
        x_padded = np.hstack((np.ones(1), x))
        # Compute weighted sum
        z_padd = np.dot(x_padded, W.T)

        np.testing.assert_array_equal(z_padd, z_add)
        expected = np.array([3.1, 3.2, 3.5, 3.2, 4])
        np.testing.assert_array_equal(z_padd, expected)

    def test_backprop(self):
        learning_rate = 0.001
        x = [1, 2]
        pt_layer = nn.Linear(in_features=len(x), out_features=3)
        with torch.no_grad():
            pt_layer.weight.copy_(torch.tensor(
                [
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.5],
                ]
            ))
            pt_layer.bias.copy_(torch.tensor([0.5, 0.5, 0.5]))
        pt_initial_weights = pt_layer.weight.detach().tolist()
        pt_output = pt_layer(torch.tensor(x, dtype=torch.float32))
        pt_result: List[float] = pt_output.tolist()
        assert len(pt_result) == 3

        optimizer = torch.optim.SGD(pt_layer.parameters(), lr=learning_rate)

        pt_loss = mse_loss(pt_output, torch.tensor([0.0, 1.0, 0.0]))
        loss = mean_squared_error_loss_categorical(pt_result, 2)
        self.assertEqual(loss, pt_loss.item())

        pt_loss.backward()
        optimizer.step()
        print(loss)

        pt_final_weights = pt_layer.weight.detach().tolist()

        layer = LinearLayer(
            neurons=3,
            inputs=len(x),
            activation_function=relu,
            derivative_activation_function=relu_derivative,
            derivative_cost_function=derivative_mean_squared_error_loss_categorical,
            test_mode=True
        )
        result = layer.forward(x)
        self.assertListEqual(pt_result, result)

        initial_weights = layer.W
        layer.backprop(x, learning_rate, label=2, print_info=True)
        final_weights = layer.W

        print(f"{pt_initial_weights=}\n{pt_final_weights=}")
        print(f"{initial_weights=}\n{final_weights=}")


if __name__ == '__main__':
    unittest.main()
