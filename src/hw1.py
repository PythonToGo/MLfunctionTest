import numpy as np

def compute_linear_fit(n=1000):
    x = np.random.uniform(0, 1, n)
    e = np.random.normal(0, np.sqrt(0.25), n)
    y = x + e
    a = np.sum(x * y) / np.sum(x**2)
    return x, y, a

def compute_polynomial_fit(n=1000):
    x = np.random.uniform(0, 1, n)
    xi = np.random.normal(0, np.sqrt(0.01), n)
    y = 30*(x - 0.25)**2*(x - 0.75)**2 + xi
    X = np.stack([x**i for i in range(5)], axis=1)
    a = np.linalg.inv(X.T @ X) @ X.T @ y
    return x, y, a
