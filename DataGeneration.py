import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
    random_data: np.array = np.random.normal(0, 1, T)
    scaled_data = (random_data - np.min(random_data)) / (np.max(random_data) - np.min(random_data))
    scaled_data = scaled_data * (max_val - min_val) + min_val
    missing_mask: np.array = np.random.rand(T) < missing_prob
    scaled_data[missing_mask] = np.nan
    return scaled_data


def generate_multiple_series(T: int, n: int, min_val: float, max_val: float, out: str) -> list:
    """
    Generates multiple time series, makes a list of all of them and writes them in a csv file
    """
    series: list = []
    
    arr = np.zeros((n, T))
    for i in range(n):
        p: float = np.random.uniform(0, 0.1)
        phi = np.random.uniform(0.7, 1)
        ser = gen_AR1__missing(phi, T, p)
        arr[i] = ser

    scaled = scale_array(arr, min_val, max_val)
    print
    with open(out, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(n):
            ser = scaled[i]
            series.append(ser)
            writer.writerow(ser)

    return series

def read_data(input: str) -> list:
    """
    Reads multiple time series from a csv file
    """
    time_series_list: list = []
    with open(input, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            time_series_list.append(np.array(row, dtype=float))

    return time_series_list

def scale_array(arr, a, b):

    non_nan_mask = ~np.isnan(arr)

    min_val = np.min(arr[non_nan_mask])
    max_val = np.max(arr[non_nan_mask])

    scaled_non_nan = a + (arr[non_nan_mask] - min_val) * (b - a) / (max_val - min_val)

    scaled_arr = np.empty_like(arr)
    scaled_arr[non_nan_mask] = scaled_non_nan
    scaled_arr[np.isnan(arr)] = np.nan

    return scaled_arr



def gen_AR1(phi, samples):
    ar_process = sm.tsa.ArmaProcess(ar=np.r_[1, -phi], ma=np.array([1]))
    ar_series = ar_process.generate_sample(nsample=samples)
    return ar_series

def gen_AR1__missing(phi, samples, p):
    ar_series = gen_AR1(phi, samples)
    missing_mask = np.random.rand(samples) < p
    ar_series[missing_mask] = np.nan
    return ar_series 