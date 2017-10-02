
import numpy as np


def soft_threshold_odd(a, lambd, rho):

    # The off-diagonal Lasso penalty function.
    # Computes the Z0 update with off-diagonal
    # soft-threshold operator

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

    # The element-wise l1 penalty function.
    # Used in (Z1, Z2) update

    eta = 2*beta/rho
    dimension = np.shape(a)[0]
    e = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            if abs(a[i, j]) > eta:
                e[i, j] = np.sign(a[i, j])*(
                    abs(a[i, j]) - eta)
    return e


def group_lasso(a, beta, rho):

    # The Group Lasso l2 penalty function.
    # Used in (Z1, Z2) update

    eta = 2*beta/rho
    dimension = np.shape(a)[0]
    e = np.zeros((dimension, dimension))
    for j in range(dimension):
        l2_norm = np.linalg.norm(a[:, j])
        if l2_norm > eta:
            e[:, j] = (1 - eta/l2_norm)*a[:, j]
    return e


def perturbed_node(theta_pre, theta, u1, u2, beta, rho, ct, cc):

    # The row-column overlap penalty function (Perturbed Node).
    # Used in (Z1, Z2 update)
    # Computes the update as a sub-ADMM algorithm,
    # as no closed form update exists.

    """ Initialize ADMM algorithm """
    dimension = np.shape(theta)[0]
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
        v = group_lasso(a, beta, rho)

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
