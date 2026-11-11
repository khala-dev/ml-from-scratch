import numpy as np


def predict(theta0, theta1, x):
    """
    hypothesis: yhat = theta0 + theta1 * x
    """
    return theta0 + theta1 * x


def cost(theta0, theta1, x, y):
    n = len(x)

    y_hat = predict(theta0, theta1, x)
    errors = y_hat - y
    squared_errors = errors**2
    return squared_errors.sum() / (2 * n)


def gradient_descent(x, y, alpha, n_iterations, theta0=0.0, theta1=0.0, tolerance=None):
    n = len(x)
    cost_history = []

    for _ in range(n_iterations):
        errors = predict(theta0, theta1, x) - y

        grad0 = errors.sum() / n
        grad1 = (errors * x).sum() / n

        theta0 = theta0 - alpha * grad0
        theta1 = theta1 - alpha * grad1

        cost_history.append(cost(theta0, theta1, x, y))

        if tolerance is not None and len(cost_history) > 1:
            if abs(cost_history[-2] - cost_history[-1]) < tolerance:
                break

    return theta0, theta1, cost_history


def normal_equation(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta[0], theta[1]
