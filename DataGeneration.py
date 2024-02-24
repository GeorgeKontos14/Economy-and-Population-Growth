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