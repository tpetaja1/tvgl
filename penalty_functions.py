
import numpy as np


def soft_threshold_odd(a, lambd, rho):
    parameter = lambd/rho
    dimension = np.shape(a)[0]
    e = np.eye(dimension, dimension)
    for i in range(dimension - 1):
        for j in range(i + 1, dimension):
            if abs(a[i, j]) > parameter:
                result = np.sign(a[i, j])*(
                    abs(a[i, j]) - parameter)
                e[i, j] = result
                e[j, i] = result
    return e


def soft_threshold_odds(aa, lambd, rho):
    parameter = lambd/rho
    dimension = np.shape(aa[0])[0]
    ee = [np.eye(dimension, dimension) for i in range(len(aa))]
    for a, e in zip(aa, ee):
        for i in range(dimension - 1):
            for j in range(i + 1, dimension):
                if abs(a[i, j]) > parameter:
                    result = np.sign(a[i, j])*(
                        abs(a[i, j]) - parameter)
                    e[i, j] = result
                    e[j, i] = result
    return ee


def group_lasso_penalty(a, beta, rho):
    nju = 2*beta/rho
    dimension = np.shape(a)[0]
    e = np.zeros((dimension, dimension))
    for j in range(dimension):
        l2_norm = np.linalg.norm(a[:, j])
        if l2_norm > nju:
            e[:, j] = (1 - nju/l2_norm)*a[:, j]
    return e


def group_lasso_penaltys(aa, beta, rho):
    nju = 2*beta/rho
    dimension = np.shape(aa[0])[0]
    ee = [np.zeros((dimension, dimension)) for i in range(len(aa))]
    for a, e in zip(aa, ee):
        for j in range(dimension):
            l2_norm = np.linalg.norm(a[:, j])
            if l2_norm > nju:
                e[:, j] = (1 - nju/l2_norm)*a[:, j]
    return ee
