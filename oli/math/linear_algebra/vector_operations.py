import unittest
from typing import List

from oli.math.math_utility import pretty_print_matrix


def dot_product(a: List[float], b: List[float]) -> float:
    """
    Compute the dot product of two vectors.
    :param a: First vector.
    :param b: Second vector.
    :return: Dot product of vector a and b.
    """
    result = 0
    for i in range(0, len(a)):
        result += a[i] * b[i]
    return result


def matrix_vector_multiplication(A: List[List[float]], v: List[float]) -> List[float]:
    assert isinstance(A, List)
    assert isinstance(A[0], List)
    assert isinstance(v, List)
    assert isinstance(v[0], float) or isinstance(v[0], int)

    if len(A) != len(v):
        pretty_print_matrix(A, label="A:")
        pretty_print_matrix(v, label="v:")
        raise Exception(
            f"Multiplication is only possible if the number of columns of A corresponds to the number of items in the vector. {len(A)=} {len(v)=}")

    Av: List[float] = [0 for i in range(len(A))]
    for neuron_index in range(len(A)):
        sum = 0
        for i in range(len(v)):
            mult = A[neuron_index][i] * v[i]
            sum += mult
        Av[neuron_index] = sum

    return Av


class TestMatrixVectorMultiplication(unittest.TestCase):
    def test_matrix_vector(self):
        A = [
            [5, 1, 3],
            [1, 1, 1],
            [1, 2, 1]
        ]
        v = [
            1, 2, 3
        ]
        expected = [16, 6, 8]
        self.assertEqual(matrix_vector_multiplication(A, v), expected)


if __name__ == '__main__':
    unittest.main()
