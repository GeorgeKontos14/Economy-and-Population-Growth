import numpy as np
import matplotlib.pyplot as plt
import csv

def generate_continuous_data(T: int, min_val: float, max_val: float) -> np.array :
    """
    Generatres a time series of length T, with values between min_val and max_val
    """
    random_data: np.array = np.random.rand(T)
    return random_data * (max_val - min_val) + min_val

def generate_unbalanced_data(T: int, min_val: float, max_val: float, missing_prob: float) -> np.array:
    """
    Generates a time series with missing values
    """
    random_data: np.array = np.random.rand(T)
    missing_mask: np.array = np.random.rand(T) < missing_prob
    random_data[missing_mask] = np.nan
    return random_data * (max_val - min_val) + min_val

def plot_logarithm(T: int, y: np.array):
    """
    For a given time series y that might have missing values,
    plots its logarithm
    """
    plt.figure(figsize=(25,10))

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

def generate_multiple_series(T: int, n: int, min_val: float, max_val: float, out: str) -> list:
    """
    Generates multiple time series, makes a list of all of them and writes them in a csv file
    """
    # output = open(out, "w")
    series: list = []
    # for i in range(n):
    #     p: float = np.random.uniform(0, 0.1)
    #     y = generate_unbalanced_data(T, min_val, max_val, p)
    #     series.append(y)
    #     output.writerow(','.join(map(str, y)))
    
    with open(out, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(n):
            p: float = np.random.uniform(0, 0.1)
            y = generate_unbalanced_data(T, min_val, max_val, p)
            series.append(y)
            writer.writerow(y)

    return series

def read_data(input: str) -> list:
    """
    Reads multiple time series from a csv file
    """
    time_series_list: list = []
    with open(input, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            time_series_list.append(row)

    return time_series_list

def plot_multiple_series(series_list: list):
    """
    Given a list of time series, plots the logarithms of the series in the same plot
    """
    plt.figure(figsize=(25,10))

    n: int = len(series_list)
    T: int = len(series_list[0])
    colors = plt.cm.jet(np.linspace(0, 1, n))
    for i, series in enumerate(series_list):
        start_idx: int = 0
        log_series: np.array = np.log(series)
        for j in range(T):
            if np.isnan(log_series[j]):
                if start_idx < j:
                    plt.plot(range(start_idx, j), log_series[start_idx:j], color=colors[i], linestyle='-')
                start_idx = j + 1
        if start_idx < T:
            plt.plot(range(start_idx, T), log_series[start_idx:], color=colors[i], linestyle='-')