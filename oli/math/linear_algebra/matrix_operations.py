def transpose(A: list[list[float]]) -> list[list[float]]:
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


def multiplication(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Function to multiply two 2d matrices.
    :param A: First matrix.
    :param B: Second matrix.
    :return: Matrix product.
    """
    if len(A) != len(B[0]):
        raise Exception(
            "Multiplication is only possible if the number of columns of A corresponds to the number of rows in B.")
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


def frobenius_inner_product(A: list[list[float]], B: list[list[float]]) -> float:
    result = 0
    for m in range(0, len(A)):
        for j in range(0, len(A[0])):
            result += A[m][j] * B[m][j]
            print(f"m: {m}, j: {j}, A[m][j]: {A[m][j]}, B[m][j]: {B[m][j]}\t\tA[m][j] * B[m][j]: {A[m][j] * B[m][j]}")
    return result
