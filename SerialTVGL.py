
from TVGL import TVGL
import penalty_functions as pf
import numpy as np


class SerialTVGL(TVGL):

    def __init__(self, *args, **kwargs):
        super(SerialTVGL, self).__init__(processes=1, *args, **kwargs)

    def theta_update(self):
        for i in range(self.blocks):
            a = (self.z0s[i] + self.z1s[i] + self.z2s[i] -
                 self.u0s[i] - self.u1s[i] - self.u2s[i])/3
            at = a.transpose()
            m = self.nju*(a + at)/2 - self.emp_cov_mat[i]
            d, q = np.linalg.eig(m)
            qt = q.transpose()
            sqrt_matrix = np.sqrt(d**2 + 4/self.nju*np.ones(self.dimension))
            diagonal = np.diag(d) + np.diag(sqrt_matrix)
            self.thetas[i] = np.real(
                self.nju/2*np.dot(np.dot(q, diagonal), qt))

    def z_update(self):
        self.z0_update()
        self.z1_z2_update()

    def z0_update(self):
        self.z0s = [pf.soft_threshold_odd(
            self.thetas[i] + self.u0s[i], self.lambd, self.rho)
                    for i in range(self.blocks)]

    def z1_z2_update(self):
        if self.penalty_function == "perturbed_node":
            for i in range(1, self.blocks):
                self.z1s[i-1], self.z2s[i] = pf.perturbed_node(self.thetas[i-1],
                                                               self.thetas[i],
                                                               self.u1s[i-1],
                                                               self.u2s[i],
                                                               self.beta,
                                                               self.rho)
        else:
            aa = [self.thetas[i] - self.thetas[i-1] + self.u2s[i] - self.u1s[i-1]
                  for i in range(1, self.blocks)]
            ee = [getattr(pf, self.penalty_function)(a, self.beta, self.rho) for a in aa]
            for i in range(1, self.blocks):
                summ = self.thetas[i-1] + self.thetas[i] + self.u1s[i-1] + self.u2s[i]
                self.z1s[i-1] = 0.5*(summ - ee[i-1])
                self.z2s[i] = 0.5*(summ + ee[i-1])

    def u_update(self):
        for i in range(self.blocks):
            self.u0s[i] = self.u0s[i] + self.thetas[i] - self.z0s[i]
        for i in range(1, self.blocks):
            self.u2s[i] = self.u2s[i] + self.thetas[i] - self.z2s[i]
            self.u1s[i-1] = self.u1s[i-1] + self.thetas[i-1] - self.z1s[i-1]
