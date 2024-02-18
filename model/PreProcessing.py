import numpy as np

def remove_isolated_values(y: np.array) -> np.array:
    """
    Removes the isolated values from a time series
    """
    # TODO: Optimization
    res: np.array = np.copy(y)
    if y[1] is None:
        res[0] = None
    for i in range(1, len(y)-1):
        if y[i] is None:
            continue
        if y[i-1] is None and y[i+1] is None:
            res[i] = None
    if y[len(y)-2] is None:
        res[len(y)-1] = None
    return res
