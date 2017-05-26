import os
import pathlib
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

try:
    import seaborn
    seaborn.set()
except ImportError:
    pass


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(theta, x, y):
    "Compute cost for logistic regression"
    m = len(x)
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    h = sigmoid(x * theta.T)

    log_l = (
        np.multiply(-y, np.log(h)) -
        np.multiply(1 - y, np.log(1 - h))
    )

    return log_l.sum() / m


def gradient(theta, x, y):
    "Compute gradient for logistic regression"
    m = len(x)
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    h = sigmoid(x * theta.T)
    error = h - y
    grad = (x.T * error) / m
    return grad


def predict(theta, x):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters theta
    """
    h = sigmoid(x * theta.T)
    return h >= 0.5


def plot_data(data):
    """
    Plotting data with + indicating (y = 1) examples
    and o indicating (y = 0) examples
    """
    positive = data.where(data['Admitted'] == 1)
    negative = data.where(data['Admitted'] == 0)

    _, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')

    plt.show()


def _load_data(filepath):
    data = pd.read_csv(filepath, header=None,  names=['Exam 1', 'Exam 2', 'Admitted'])
    # plot_data(data)
    data.insert(0, 'Ones', 1)
    x = data.iloc[:, 0:3]
    y = data.iloc[:, 3:4]
    return data, np.array(x.values), np.array(y.values)


def main():
    data, x, y = _load_data(pathlib.Path.cwd() / os.path.dirname(__file__) / 'ex2data1.txt')
    # initialize fitting parameters
    theta = np.zeros(3)

    # compute and display initial cost and gradient
    cost = compute_cost(theta, x, y)
    grad = gradient(theta, x, y)
    print(f'Cost at initial theta (zeros): {cost} \nExpected cost (approx): 0.693\n')
    print(f'Gradient at initial theta (zeros): {grad.flat[0]}, {grad.flat[1]}, {grad.flat[2]}\n')
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

    # compute and display cost and gradient with non-zero theta
    test_theta = np.array([-24, 0.2, 0.2])
    cost = compute_cost(test_theta, x, y)
    grad = gradient(test_theta, x, y)
    print(f'\nCost at test theta: {cost}\nExpected cost (approx): 0.218\n')
    print(f'Gradient at test theta: \n{grad.flat[0]}, {grad.flat[1]}, {grad.flat[2]}')
    print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

    # Run fmin_tnc to obtain the optimal theta
    result = opt.fmin_tnc(
        func=compute_cost,
        x0=theta,
        fprime=gradient,
        args=(x, y)
    )
    theta_min = np.matrix(result[0])
    cost = compute_cost(theta_min, x, y)
    print(f'Cost at theta found by fmin_tnc: {cost}\n Expected cost (approx): 0.203\n')
    print('theta: \n', theta_min, '\nExpected theta (approx):\n-25.161\n 0.206\n 0.201\n')

    # Predict probability for a student with score 45 on exam 1
    # and score 85 on exam 2
    prob = sigmoid(np.array([1, 45, 85]) * theta_min.T).flat[0]
    print(f'For a student with scores 45 and 85, we predict an admission probability of {prob}')
    print('Expected value: 0.775 +/- 0.002\n')

    # Compute accuracy on our training set
    p = predict(theta_min, x)
    accuracy = (y[np.where(p == y)].size / y.size) * 100
    print(f'Train Accuracy: {accuracy}')
    print('Expected accuracy (approx): 89.0')


if __name__ == '__main__':
    main()
