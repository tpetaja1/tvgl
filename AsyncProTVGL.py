
import numpy as np
import multiprocessing
import mp_workers2 as mp
from TVGL import TVGL


class AsyncProTVGL(TVGL):

    def __init__(self, filename, blocks, processes=1):
        super(AsyncProTVGL, self).__init__(filename, blocks, processes)

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
        pool = multiprocessing.Pool(self.processes)
        res_z0s = pool.apply_async(mp.z0_update,
                                   (self.thetas, self.z0s,
                                    self.u0s, self.lambd,
                                    self.rho, self.blocks))
        res_z1z2s = pool.apply_async(mp.z1_z2_update,
                                     (self.thetas, self.z1s, self.z2s,
                                      self.u1s, self.u2s, self.beta,
                                      self.rho, self.blocks))
        self.z0s = res_z0s.get()
        z1s_z2s = res_z1z2s.get()
        self.z1s = z1s_z2s[0]
        self.z2s = z1s_z2s[1]
        pool.close()

    def u_update(self):
        pool = multiprocessing.Pool(self.processes)
        res_u0s = pool.apply_async(mp.u0_update,
                                   (self.u0s, self.thetas,
                                    self.z0s, self.blocks))
        res_u1s = pool.apply_async(mp.u1_update,
                                   (self.u1s, self.thetas,
                                    self.z1s, self.blocks))
        res_u2s = pool.apply_async(mp.u2_update,
                                   (self.u2s, self.thetas,
                                    self.z2s, self.blocks))
        self.u0s = res_u0s.get()
        self.u1s = res_u1s.get()
        self.u2s = res_u2s.get()
        pool.close()
