def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]
