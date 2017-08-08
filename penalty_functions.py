
import numpy as np


def soft_threshold_odd(a, lambd, rho):
    dimension = np.shape(a)[0]
    e = np.ones((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            if i != j:
                if abs(a[i, j]) <= lambd/rho:
                    e[i, j] = 0
                else:
                    e[i, j] = np.sign(a[i, j])*(
                        abs(a[i, j]) - lambd/rho)
    return e


def group_lasso_penalty(a, nju):
    dimension = np.shape(a)[0]
    e = np.zeros((dimension, dimension))
    for j in range(dimension):
        l2_norm = np.linalg.norm(a[:, j])
        if l2_norm <= nju:
            e[:, j] = np.zeros(dimension)
        else:
            e[:, j] = (1 - nju/l2_norm)*a[:, j]
    return e
