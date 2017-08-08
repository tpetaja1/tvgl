
import penalty_functions as pf
import numpy as np
import traceback


def theta_update((theta, z0, z1, z2, u0, u1, u2, emp_cov_mat, nju)):
    try:
        dimension = np.shape(theta)[0]
        a = (z0 + z1 + z2 - u0 - u1 - u2)/3
        at = a.transpose()
        m = nju*(a + at)/2 - emp_cov_mat
        d, q = np.linalg.eig(m)
        qt = q.transpose()
        sqrt_matrix = np.sqrt(d**2 + 4/nju*np.ones(dimension))
        diagonal = np.diag(d) + np.diag(sqrt_matrix)
        theta_new = np.real(
            nju/2*np.dot(np.dot(q, diagonal), qt))
    except Exception as e:
        traceback.print_exc()
        raise e
    return theta_new


def z0_update((theta, u0, lambd, rho)):
    try:
        z0 = pf.soft_threshold_odd(theta + u0, lambd, rho)
    except Exception as e:
        traceback.print_exc()
        raise e
    return z0


def z1_z2_update((theta, theta_pre, u1, u1_pre, u2, beta, rho)):
    try:
        a = theta - theta_pre + u2 - u1_pre
        e = pf.group_lasso_penalty(a, 2*beta/rho)
        z1 = 0.5*(theta_pre + theta + u1 + u2) - 0.5*e
        z2 = 0.5*(theta_pre + theta + u1 + u2) + 0.5*e
    except Exception as e:
        traceback.print_exc()
        raise e
    return (z1, z2)


def u0_update((theta, u0, z0)):
    try:
        u0 = u0 + theta - z0
    except Exception as e:
        traceback.print_exc()
        raise e
    return u0


def u1_u2_update((theta, theta_pre, u1_pre, u2, z1_pre, z2)):
    try:
        u1 = u1_pre + theta_pre - z1_pre
        u2 = u2 + theta - z2
    except Exception as e:
        traceback.print_exc()
        raise e
    return (u1, u2)


def u1_update((theta, u1, z1)):
    try:
        u1 = u1 + theta - z1
    except Exception as e:
        traceback.print_exc()
        raise e
    return u1


def u2_update((theta, u2, z2)):
    try:
        u2 = u2 + theta - z2
    except Exception as e:
        traceback.print_exc()
        raise e
    return u2
