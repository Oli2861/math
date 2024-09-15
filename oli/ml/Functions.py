def softmax(x: list[float]) -> list[float]:
    """
    Converts vector into probability distribution of outcomes.
    :param x: Vector.
    :return: Probability distribution of outcomes.
    """
    sum = 0.0000000000000000000000001
    for item in x:
        sum += item
    return [curr / sum for curr in x]
