import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ex1 import compute_cost, gradient_descent


def feature_normalize(x):
    return (x - x.mean()) / x.std()


def _load_data(filepath):
    r = pd.read_csv(filepath, header=None,  names=['Size', 'Bedrooms', 'Price'])
    # normalizing features
    data = feature_normalize(r)

    # add a column of ones to x
    data.insert(0, 'Ones', 1)

    x_values = data.iloc[:, 0:3].values
    y_values = r.iloc[:, 2:3].values

    return data, np.matrix(x_values), np.matrix(y_values), r.mean(), r.std()


def _plot_convergence_graph(*, j_history=None, num_iters=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(num_iters), j_history, 'r')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost Function')
    ax.set_title('Convergence graph')
    plt.show()


def main():
    data, x, y, mu, std = _load_data(pathlib.Path.cwd() / os.path.dirname(__file__) / 'ex1data2.txt')
    # plot_data(data)
    theta = np.matrix(np.array([0, 0, 0]))

    # some gradient descent settings
    alpha = 0.01
    num_iters = 400

    # perform linear regression
    theta, j_history = gradient_descent(x, y, theta, alpha, num_iters)

    _plot_convergence_graph(j_history=j_history, num_iters=num_iters)

    cost = compute_cost(x, y, theta)

    print(f'Theta computed from gradient descent: \n{theta}, \ncost = {cost}\n')

    sample1 = [
        1,
        (1650 - mu['Size']) / std['Size'],
        (3 - mu['Bedrooms']) / std['Bedrooms']
    ]
    predict = sample1 * theta.T
    price = predict.item()
    print(f'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): {price}\n')


if __name__ == '__main__':
    main()
