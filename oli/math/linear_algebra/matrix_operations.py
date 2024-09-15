import unittest
from typing import List

from oli.math.math_utility import pretty_print_matrix


def transpose(A: List[List[float]]) -> List[List[float]]:
    """
    Function to transpose 2d matrices / 1d vectors.
    :param A: Matrix / vector to be transposed.
    :return: Transposed matrix / vector.
    """
    result = []
    for rowIndex, row in enumerate(A):
        for columnIndex, columnEntry in enumerate(row):
            # Original columnIndex = row index of new matrix
            if rowIndex == 0:
                result.append([])
            result[columnIndex].append(columnEntry)
    return result


def multiplication(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Function to multiply two 2d matrices.
    :param A: First matrix.
    :param B: Second matrix.
    :return: Matrix product.
    """
    assert isinstance(A, List)
    assert isinstance(A[0], List)
    assert isinstance(B, List)
    assert isinstance(B[0], List)

    if len(A) != len(B[0]):
        print(A)
        print(B)
        raise Exception(
            f"Multiplication is only possible if the number of columns of A corresponds to the number of rows in B. {len(A)=} {len(B[0])=}")

    m = len(A)  # Rows of A
    n = len(A[0])  # Columns of A
    n = len(B)  # Rows of B
    p = len(B[0])  # Columns of B
    C = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(0, m):
        for j in range(0, p):
            C[i][j] = 0
            for u in range(0, n):
                a_iu = A[i][u]
                b_uj = B[u][j]
                print(f"i: {i}, j: {j}, u: {u}, a_iu: {a_iu}, b_uj: {b_uj}")
                C[i][j] += a_iu * b_uj

    return C


def frobenius_inner_product(A: List[List[float]], B: List[List[float]]) -> float:
    result = 0
    for m in range(0, len(A)):
        for j in range(0, len(A[0])):
            result += A[m][j] * B[m][j]
            print(f"m: {m}, j: {j}, A[m][j]: {A[m][j]}, B[m][j]: {B[m][j]}\t\tA[m][j] * B[m][j]: {A[m][j] * B[m][j]}")
    return result


class TestMatrixMultiplication(unittest.TestCase):

    def test_square_matrices(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        expected = [[19, 22], [43, 50]]
        self.assertEqual(multiplication(A, B), expected)

    def test_rectangular_matrices(self):
        A = [[1, 2, 3], [4, 5, 6]]
        B = [[7, 8], [9, 10], [11, 12]]
        expected = [[58, 64], [139, 154]]
        self.assertEqual(multiplication(A, B), expected)

    def test_zeros_matrix(self):
        A = [[0, 0], [0, 0]]
        B = [[0, 0], [0, 0]]
        expected = [[0, 0], [0, 0]]
        self.assertEqual(multiplication(A, B), expected)

    def test_ones_matrix(self):
        A = [[1, 1], [1, 1]]
        B = [[1, 1], [1, 1]]
        expected = [[2, 2], [2, 2]]
        self.assertEqual(multiplication(A, B), expected)

    def test_negative_numbers(self):
        A = [[-1, 1], [-2, 2]]
        B = [[-3, 3], [-4, 4]]
        expected = [[-1, 1], [-2, 2]]
        self.assertEqual(multiplication(A, B), expected)


if __name__ == '__main__':
    unittest.main()
