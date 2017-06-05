import os
import pathlib
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from ex2 import sigmoid, predict

try:
    import seaborn
    seaborn.set()
except ImportError:
    pass


def plot_data(data):
    """
    Plotting data with + indicating (y = 1) examples
    and o indicating (y = 0) examples
    """
    positive = data.where(data['Accepted'] == 1)
    negative = data.where(data['Accepted'] == 0)

    _, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(positive['Microchip Test 1'], positive['Microchip Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Microchip Test 1'], negative['Microchip Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set(xlabel='Test 1 Score', ylabel='Test 2 Score', title='Microship training set')
    plt.show()


def map_feature(data, *, degree=6):
    """
    Maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    x1, x2, x1.^2, x2.^2, x1*x2, x1*x2.^2, etc..
    inputs x1, x2 must be the same size.
    """
    x1 = data['Microchip Test 1']
    x2 = data['Microchip Test 2']
    data.insert(3, 'Ones', 1)

    for i in range(1, degree):
        for j in range(i):
            data[f'F{i}{j}'] = np.power(x1, i - j) * np.power(x2, j)

    data.drop('Microchip Test 1', axis=1, inplace=True)
    data.drop('Microchip Test 2', axis=1, inplace=True)


def _load_data(filepath):
    data = pd.read_csv(filepath, header=None,  names=['Microchip Test 1', 'Microchip Test 2', 'Accepted'])
    # Note that map_feature also adds a column of ones for us, so the intercept
    # term is handled
    map_feature(data)

    # plot_data(data)
    num_cols = data.shape[1]
    x = data.iloc[:, 1:num_cols]
    y = data.iloc[:, 0:1]
    return data, np.array(x.values), np.array(y.values)


def cost_reg(theta, x, y, lambda_):
    "Compute cost for logistic regression with regularization"
    m = len(x)
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    h = sigmoid(x * theta.T)

    log_l = np.multiply(-y, np.log(h)) - np.multiply(1 - y, np.log(1 - h))

    reg = (lambda_ / 2 * m) * np.power(theta[:, 1:theta.shape[1]], 2).sum()
    return log_l.sum() / m + reg


def gradient_reg(theta, x, y, lambda_):
    "Compute gradient for logistic regression with regularization"
    m = len(x)
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    h = sigmoid(x * theta.T)
    error = h - y

    for i in range(parameters):
        g = (error.T * x[:, i]) / m
        grad[i] = g if (i == 0) else g + ((lambda_ / m) * theta[:, i])

    return grad


def main():
    data, x, y = _load_data(pathlib.Path.cwd() / os.path.dirname(__file__) / 'ex2data2.txt')
    # initialize fitting parameters
    theta = np.zeros(x.shape[1])

    # set regularization parameter lambda to 1
    lambda_ = 1
    cost = cost_reg(theta, x, y, lambda_)
    grad = gradient_reg(theta, x, y, lambda_)

    print(f'Cost at initial theta (zeros): {cost}\nExpected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros) - first five values only:\n', grad[:5])
    print('Expected gradients (approx) - first five values only:\n')
    print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

    result = opt.fmin_tnc(func=cost_reg, x0=theta, fprime=gradient_reg, args=(x, y, lambda_))
    theta_min = np.matrix(result[0])
    p = predict(theta_min, x)
    accuracy = (y[np.where(p == y)].size / y.size) * 100

    print(f'Train Accuracy: {accuracy}')
    print('Expected accuracy (with lambda = 1): 83.1 (approx)')


if __name__ == '__main__':
    main()
