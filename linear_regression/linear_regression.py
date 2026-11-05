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
    squared_errors = errors ** 2
    return squared_errors.sum() / (2 * n)

