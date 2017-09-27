
import numpy as np


def soft_threshold_odd(a, lambd, rho):
    parameter = lambd/rho
    dimension = np.shape(a)[0]
    e = np.eye(dimension)
    for i in range(dimension - 1):
        for j in range(i + 1, dimension):
            if abs(a[i, j]) > parameter:
                result = np.sign(a[i, j])*(
                    abs(a[i, j]) - parameter)
                e[i, j] = result
                e[j, i] = result
    return e


def element_wise(a, beta, rho):
    nju = 2*beta/rho
    dimension = np.shape(a)[0]
    e = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            if abs(a[i, j]) > nju:
                e[i, j] = np.sign(a[i, j])*(
                    abs(a[i, j]) - nju)
    return e


def group_lasso(a, beta, rho):
    nju = 2*beta/rho
    dimension = np.shape(a)[0]
    e = np.zeros((dimension, dimension))
    for j in range(dimension):
        l2_norm = np.linalg.norm(a[:, j])
        if l2_norm > nju:
            e[:, j] = (1 - nju/l2_norm)*a[:, j]
    return e


def perturbed_node(theta_pre, theta, u1, u2, beta, rho, ct, cc):

    """ Initialize ADMM algorithm """
    dimension = np.shape(theta)[0]
    nju = beta/(2*rho)
    y1 = np.ones((dimension, dimension))
    y2 = np.ones((dimension, dimension))
    v = np.ones((dimension, dimension))
    w = np.ones((dimension, dimension))
    uu1 = np.zeros((dimension, dimension))
    uu2 = np.zeros((dimension, dimension))

    """ Run algorithm """
    iteration = 0
    y_pre = []
    stopping_criteria = False
    while iteration < 1000 and stopping_criteria is False:
        a = (y1 - y2 - w - uu1 + (w.transpose() - uu2).transpose())/2

        """ V Update """
        e = np.zeros((dimension, dimension))
        for j in range(dimension):
            l2_norm = np.linalg.norm(a[:, j])
            if l2_norm > nju:
                e[:, j] = (1 - nju/l2_norm)*a[:, j]
        v = e

        """ W, Y1, Y2 Update """
        b = np.zeros((3*dimension, dimension))
        b[0:dimension, :] = (v + uu2).transpose()
        b[dimension:2*dimension, :] = theta_pre + u1
        b[2*dimension:3*dimension, :] = theta + u2
        d = v + uu1
        wyy = np.dot(cc, 2*b - np.dot(ct, d))
        w = wyy[0:dimension, :]
        y1 = wyy[dimension:2*dimension, :]
        y2 = wyy[2*dimension:3*dimension, :]

        """ UU1, UU2 Update """
        uu1 = uu1 + (v + w) - (y1 - y2)
        uu2 = uu2 + v - w.transpose()

        """ Check stopping criteria """
        if iteration > 0:
            dif = y1 - y_pre
            fro_norm = np.linalg.norm(dif)
            if fro_norm < 1e-3:
                stopping_criteria = True
        y_pre = list(y1)
        iteration += 1
    return (y1, y2)
