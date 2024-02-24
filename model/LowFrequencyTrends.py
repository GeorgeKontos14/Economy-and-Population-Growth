import numpy as np
import statsmodels.api as sm
from model import Regressors
from model import PreProcessing

def find_trends(yi: np.array, cutoff:float):
    """
    Given a time series y, it finds the necessary regressors, calculates their coefficients
    and projects the result onto the base line low frequency trend.
    """
    # This method ONLY works for balanced data
    T = len(yi)
    y: np.array = PreProcessing.remove_isolated_values(yi)

    regressors: np.ndarray = Regressors.find_time_series_regressors(y, cutoff).T
    r_with_const = sm.add_constant(regressors)
    model = sm.OLS(y, r_with_const)
    results = model.fit()
    return results.params, results.predict()