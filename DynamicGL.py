
from TVGL import TVGL
import penalty_functions as pf
import numpy as np
import multiprocessing
import time
import traceback


def mp_dynamic_gl((theta, z0, u0, emp_cov_mat, rho,
                  lambd, nju, dimension, max_iter)):
    try:
        iteration = 0
        stopping_criteria = False
        theta_pre = []
        while iteration < max_iter and stopping_criteria is False:
            """ Theta update """
            a = z0 - u0
            at = a.transpose()
            m = nju*(a + at)/2 - emp_cov_mat
            d, q = np.linalg.eig(m)
            qt = q.transpose()
            sqrt_matrix = np.sqrt(d**2 + 4/nju*np.ones(dimension))
            diagonal = np.diag(d) + np.diag(sqrt_matrix)
            theta = np.real(
                nju/2*np.dot(np.dot(q, diagonal), qt))
            """ Z-update """
            z0 = pf.soft_threshold_odd(theta + u0, lambd, rho)
            """ U-update """
            u0 += theta - z0
            """ Check stopping criteria """
            if iteration > 0:
                dif = theta - theta_pre
                fro_norm = np.linalg.norm(dif)
                if fro_norm < 1e-5:
                    stopping_criteria = True
            theta_pre = list(theta)
            iteration += 1
    except Exception as e:
        traceback.print_exc()
        raise e
    return theta


class DynamicGL(TVGL):

    def __init__(self, *args, **kwargs):
        super(DynamicGL, self).__init__(beta=0, *args, **kwargs)
        self.nju = float(self.obs)/float(self.rho)
        self.iteration = "n/a"
        self.penalty_function = "n/a"

    def get_rho(self):
        return self.obs + 1

    def run_algorithm(self, max_iter=10000):
        start_time = time.time()
        p = multiprocessing.Pool(self.processes)
        inputs = [(self.thetas[i], self.z0s[i], self.u0s[i],
                   self.emp_cov_mat[i], self.rho,
                   self.lambd, self.nju, self.dimension, max_iter)
                  for i in range(self.blocks)]
        self.thetas = p.map(mp_dynamic_gl, inputs)
        p.close()
        p.join()
        self.run_time = '{0:.3g}'.format(time.time() - start_time)
        self.thetas = [np.round(theta, self.roundup) for theta in self.thetas]
