
import numpy as np
import traceback


def mp_theta_update((theta, z0, z1, z2, u0, u1, u2, emp_cov_mat, nju)):
    try:
        dimension = np.shape(theta)[0]
        a = (z0 + z1 + z2 - u0 - u1 - u2)/3
        at = a.transpose()
        m = nju*(a + at)/2 - emp_cov_mat
        d, q = np.linalg.eig(m)
        qt = q.transpose()
        sqrt_matrix = np.sqrt(d**2 + 4/nju*np.ones(dimension))
        diagonal = np.diag(d) + np.diag(sqrt_matrix)
        theta_new = nju/2*np.dot(np.dot(q, diagonal), qt)
    except Exception as e:
        traceback.print_exc()
        raise e
    return theta_new


def mp_z0_update((theta, u0, lambd, rho)):
    try:
        z0 = soft_threshold_odd(theta + u0, lambd, rho)
    except Exception as e:
        traceback.print_exc()
        raise e
    return z0


def mp_z1_z2_update((theta, theta_pre, u1, u1_pre, u2, beta, rho)):
    try:
        a = theta - theta_pre + u2 - u1_pre
        e = group_lasso_penalty(a, 2*beta/rho)
        z1 = 0.5*(theta_pre + theta + u1 + u2) - 0.5*e
        z2 = 0.5*(theta_pre + theta + u1 + u2) + 0.5*e
    except Exception as e:
        traceback.print_exc()
        raise e
    return (z1, z2)


def mp_u0_update((theta, u0, z0)):
    try:
        u0 = u0 + theta - z0
    except Exception as e:
        traceback.print_exc()
        raise e
    return u0


def mp_u1_u2_update((theta, theta_pre, u1_pre, u2, z1_pre, z2)):
    try:
        u1 = u1_pre + theta_pre - z1_pre
        u2 = u2 + theta - z2
    except Exception as e:
        traceback.print_exc()
        raise e
    return (u1, u2)


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
