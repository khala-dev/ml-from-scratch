import numpy as np
import matplotlib.pyplot as plt
from linear_regression import gradient_descent

area = np.array([50, 65, 80, 95, 110])
price = np.array([900, 1100, 1500, 1600, 2000])

alphas = {
    0.00001: "alpha too small (slow)",
    0.0001: "alpha good (fast)",
    0.0003: "alpha too large (diverges)",
}

for alpha, label in alphas.items():
    _, _, history = gradient_descent(area, price, alpha=alpha, n_iterations=30)
    plt.plot(history, label=f"{label}: {alpha}")

plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("cost (log scale)")
plt.title("Effect of the learning rate on gradient descent")
plt.legend()
plt.savefig("learning_rate.png", dpi=150)
plt.show()