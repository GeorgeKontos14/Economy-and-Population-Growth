import numpy as np
import Regressors
import PreProcessing

def find_coefficients(yi: np.array, cutoff:float):
    """
    Given a time series y, it finds the necessary regressors, calculates their coefficients
    and projects the result onto the base line low frequency trend.
    """
    T = len(yi)
    y: np.array = PreProcessing.remove_isolated_values(yi)

    regressors: np.ndarray = Regressors.find_time_series_regressors(y, cutoff)
    Y_i, _, _ = np.linalg.lstsq(regressors, y, rcond=None)

    c_hat: float = cutoff+30/T
    X_i: np.ndarray = Regressors.find_regressors(T, c_hat)
    B, _, _ = np.linalg.lstsq(X_i, Y_i, rcond=None)

    return Y_i, B

