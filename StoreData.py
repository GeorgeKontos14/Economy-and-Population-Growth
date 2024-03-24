import csv
import numpy as np

def write_mat(mat, out):
    with open(out, mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in mat:
            vals = [float(val) for val in row]
            writer.writerow(vals)
        file.write('\n')

def write_1d(row, out):
    with open(out, mode='a', newline='') as file:
        writer = csv.writer(file)
        vals = [float(val) for val in row]
        writer.writerow(vals)   
        file.write('\n')     

def write_once(data, out):
    with open(out, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([float(val) for val in data])

def write_tuples(tuples, out):
    with open(out, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([item for tpl in tuples for item in tpl])

def write_val(data, out):
    with open(out, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data])

def clear_csv(out):
    with open(out, mode='w') as file:
        file.write('')

def read_matrices(input):
    matrices = []
    curr = []
    with open(input, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                curr.append([float(val) for val in row])
            else:
                if curr:
                    if len(curr) == 1:
                        matrices.append(np.array(curr)[0])
                    else:
                        matrices.append(np.array(curr))
                    curr = []
        
        if curr:
            matrices.append(np.array(curr))
    return matrices

def read_one_row(input):
    with open(input, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            return np.array([float(val) for val in row])
        
def read_tuples(input):
    data = []
    with open(input, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            tuples = []
            vals = [float(val) for val in row]
            for i in range(0, len(row), 3):
                tuples.append(tuple(vals[i:i+3]))
            data.append(tuples)
    return data

def read_vals(input):
    data = []
    with open(input, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(float(row[0]))
    return data