import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x, dtype=float)
    sigmoid = 1 / (1 + np.exp(-x))
    return x * sigmoid