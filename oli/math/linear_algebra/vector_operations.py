def dot_product(a: list[float], b: list[float]) -> float:
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
