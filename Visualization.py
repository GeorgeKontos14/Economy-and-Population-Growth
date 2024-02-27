import numpy as np
import matplotlib.pyplot as plt

def plot_logarithm(T: int, y: np.array):
    """
    For a given time series y that might have missing values,
    plots its logarithm
    """
    plt.figure(figsize=(20,10))

    log_data: np.array = np.log(y)
    start_idx: int = 0
    for i in range(T):
        if np.isnan(log_data[i]):
            if start_idx < i:
                plt.plot(range(start_idx, i), log_data[start_idx:i], color='blue', linestyle='-')
            start_idx = i + 1

    # Plot the last segment if it exists
    if start_idx < T:
        plt.plot(range(start_idx, T), log_data[start_idx:], color='blue', linestyle='-')

    plt.xlabel('Time')
    plt.ylabel('Logarithm of the time series')
    plt.show()

def plot_data_and_prediction(T: int, y: np.array, pred: np.array):
    """
    Given a time series and a prediction for that time series,
    plots them both on the same plot. The actual data is plotted
    in blue, while the predicted data is plotted in red.
    """
    plt.figure(figsize=(20,10))

    log_data: np.array = np.log(y)
    start_idx: int = 0
    for i in range(T):
        if np.isnan(log_data[i]):
            if start_idx < i:
                plt.plot(range(start_idx, i), log_data[start_idx:i], color='blue', linestyle='-')
            start_idx = i + 1

    # Plot the last segment if it exists
    if start_idx < T:
        plt.plot(range(start_idx, T), log_data[start_idx:], color='blue', linestyle='-')

    log_pred: np.array = np.log(pred)
    start_idx = 0
    for i in range(T):
        if np.isnan(log_data[i]):
            if start_idx < i:
                plt.plot(range(start_idx, i), log_pred[start_idx:i], color='red', linestyle='-')
            start_idx = i + 1

    # Plot the last segment if it exists
    if start_idx < T:
        plt.plot(range(start_idx, T), log_pred[start_idx:], color='red', linestyle='-')

    plt.xlabel('Time')
    plt.ylabel('Logarithm of the time series')
    plt.show()

def plot_multiple_series(series_list: list, log=True):
    """
    Given a list of time series, plots the logarithms of the series in the same plot
    """
    plt.figure(figsize=(20,10))

    n: int = len(series_list)
    T: int = len(series_list[0])
    colors = plt.cm.jet(np.linspace(0, 1, n))
    for i, series in enumerate(series_list):
        start_idx: int = 0
        log_series: np.array = series
        if log:
            log_series = np.log(series)
        for j in range(T):
            if np.isnan(log_series[j]):
                if start_idx < j:
                    plt.plot(range(start_idx, j), log_series[start_idx:j], color=colors[i], linestyle='-')
                start_idx = j + 1
        if start_idx < T:
            plt.plot(range(start_idx, T), log_series[start_idx:], color=colors[i], linestyle='-')
    
    plt.xlabel('Time')
    if log:
        plt.ylabel('Logarithm of the time series')
    else:
        plt.ylabel('GDP per Capita')
    plt.show()
    

def plot_regressors(regressors: np.ndarray, scale: float):
    plt.figure(figsize=(20,10))
    T = len(regressors[0])
    x = np.linspace(0, T, T)

    total_height = 0
    for r in regressors:
        height = max(r)-min(r)
        factor = 1
        if (height > 0):
            factor = scale/height
        plt.plot(x, r*factor-total_height, label=None)
        total_height += scale+5 # Gap for better visualization

    plt.gca().tick_params(axis='y', which='both', left=False, labelleft=False)
    plt.show()