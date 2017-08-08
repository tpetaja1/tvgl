
import numpy as np
import traceback


def z0_update(thetas, z0s, u0s, lambd, rho, blocks):
    for i in range(blocks):
        z0s[i] = soft_threshold_odd(thetas[i] + u0s[i], lambd, rho)
    return z0s


def z1_z2_update(thetas, z1s, z2s, u1s, u2s, beta, rho, blocks):
    try:
        for i in range(1, blocks):
            a = thetas[i] - thetas[i-1] + u2s[i] - u1s[i-1]
            e = group_lasso_penalty(a, 2*beta/rho)
            z1s[i-1] = 0.5*(thetas[i-1] + thetas[i]
                            + u1s[i] + u2s[i]) - 0.5*e
            z2s[i] = 0.5*(thetas[i-1] + thetas[i]
                          + u1s[i] + u2s[i]) + 0.5*e
    except Exception as e:
        traceback.print_exc()
        raise e
    return (z1s, z2s)


def u0_update(u0s, thetas, z0s, blocks):
    try:
        for i in range(blocks):
            u0s[i] = u0s[i] + thetas[i] - z0s[i]
    except Exception as e:
        traceback.print_exc()
        raise e
    return u0s


def u1_update(u1s, thetas, z1s, blocks):
    try:
        for i in range(blocks - 1):
            u1s[i] = u1s[i] + thetas[i] - z1s[i]
    except Exception as e:
        traceback.print_exc()
        raise e
    return u1s


def u2_update(u2s, thetas, z2s, blocks):
    try:
        for i in range(1, blocks):
            u2s[i] = u2s[i] + thetas[i] - z2s[i]
    except Exception as e:
        traceback.print_exc()
        raise e
    return u2s


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
