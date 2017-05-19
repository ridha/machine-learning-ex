import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cost(x, y, theta):
    """
    Computes the cost of using theta as the parameter for linear regression
    to fit the data points in x and y
    """
    m = len(x)  # number of training examples
    hypothesis = x * theta.T
    error = hypothesis - y
    sq_errors = np.power(error, 2)
    return sq_errors.sum() / (2 * m)


def gradient_descent(x, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta
    """
    theta_prev = np.matrix(np.zeros(theta.shape))
    num_features = int(theta.ravel().shape[1])
    j_history = np.zeros(num_iters)
    m = len(x)

    for i in range(num_iters):
        hypothesis = x * theta.T
        error = hypothesis - y

        for j in range(num_features):
            deriv = np.multiply(error, x[:, j]) / m
            theta_prev[0, j] = theta[0, j] - (alpha * np.sum(deriv))

        theta = theta_prev
        j_history[i] = compute_cost(x, y, theta)

    return theta, j_history


def plot_data(data, theta=None):
    " Plot the training data "
    if theta is not None:
        x = np.linspace(data.Population.min(), data.Population.max(), 100)
        f = theta[0, 0] + (theta[0, 1] * x)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(x, f, 'r', label='Prediction')
        ax.scatter(data.Population, data.Profit, label='Traning Data')
        ax.legend(loc=2)
        ax.set_xlabel('Population')
        ax.set_ylabel('Profit')
        ax.set_title('Predicted Profit vs. Population Size')
    else:
        plt.scatter(data.iloc[:, 0:1], data.iloc[:, 1:2], marker='o', c='b')
        plt.title('Profits distribution')
        plt.xlabel('Population of City in 10,000s')
        plt.ylabel('Profit in $10,000s')

    plt.show()


def _load_data(filepath):
    data = pd.read_csv(filepath, header=None, names=['Population', 'Profit'])

    # add a column of ones to x
    data.insert(0, 'Ones', 1)

    num_cols = data.shape[1]
    x_values = data.iloc[:, 0:num_cols-1].values
    y_values = data.iloc[:, num_cols-1:num_cols].values

    return data, np.matrix(x_values), np.matrix(y_values)


def main():
    data, X, y = _load_data(pathlib.Path.cwd() / (__file__.partition('.')[0] + 'data1.txt'))
    # plot_data(data)

    # initialize fitting parameters
    theta = np.matrix(np.array([0, 0]))

    j = compute_cost(X, y, theta)
    print(f'With theta = [0 ; 0], Cost computed = {j}')
    print('Expected cost value (approx) 32.07')

    # some gradient descent settings
    alpha = 0.01
    num_iters = 1500
    theta, _ = gradient_descent(X, y, theta, alpha, num_iters)

    # Y =   -3.63029144  + 1.16636235 * X1
    print('\nTheta found by gradient descent:\n', theta)
    print('Expected theta values (approx) \n  -3.6303\n  1.1664\n')
    print('Cost computed = ', compute_cost(X, y, theta))

    # predict values for population sizes of 35,000 and 70,000
    predict1 = [1, 3.5] * theta.T
    print('\nFor population = 35,000, we predict a profit of: ', predict1.item()*10000)
    predict2 = [1, 7] * theta.T
    print('For population = 70,000, we predict a profit of :', predict2.item()*10000)
    # plot_data(data, theta)


if __name__ == '__main__':
    main()
