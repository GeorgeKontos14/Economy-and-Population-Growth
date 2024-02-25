import numpy as np
import statsmodels.api as sm
from Trends import Regressors


def find_trends(yi: np.array, q: int):
    """
    Given a time series y, it finds the necessary regressors, calculates their coefficients
    and projects the result onto the base line low frequency trend.
    """
    # This method ONLY works for balanced data
    y: np.array = remove_isolated_values(yi)
    y_fixed = y.copy()
    y_fixed[np.isnan(y)] = np.nanmean(y)

    regressors: np.ndarray = Regressors.find_time_series_regressors(y, q).T
    r_with_const = sm.add_constant(regressors)
    model = sm.OLS(y_fixed, r_with_const)
    results = model.fit()
    return results.params, results.predict()

def find_baseline_trend(y: np.ndarray, q: int):
    
    regressors = Regressors.find_time_series_regressors(y, q)
    T = len(y)
    Y, _ = find_trends(y, q)
    B = np.zeros((q+16, q+1))
    baseline = Regressors.find_regressors(T, q+15).T
    b_with_const = sm.add_constant(baseline)
    
    for i, r in enumerate(regressors):
        model = sm.OLS(r, b_with_const)
        results = model.fit()
        B[:, i] = results.params

    X_model = sm.OLS(Y, B.T)
    X_results = X_model.fit()
    return X_results.params

def remove_isolated_values(y: np.array) -> np.array:
    """
    Removes the isolated values from a time series
    """
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