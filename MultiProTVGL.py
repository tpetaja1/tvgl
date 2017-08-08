
import multiprocessing
import mp_workers as mp
from TVGL import TVGL


class MultiProTVGL(TVGL):

    def __init__(self, filename, processes=1):
        super(MultiProTVGL, self).__init__(filename, processes)

    def theta_update(self):
        inputs = [(self.thetas[i], self.z0s[i], self.z1s[i], self.z2s[i],
                   self.u0s[i], self.u1s[i], self.u2s[i],
                   self.emp_cov_mat[i], self.nju)
                  for i in range(self.blocks)]
        pool = multiprocessing.Pool(self.processes)
        self.thetas = pool.map(mp.mp_theta_update, inputs)
        pool.close()
        #pool.join()

    def z_update(self):
        self.z0_update()
        self.z1_z2_update()
    
    def z0_update(self):
        inputs = [(self.thetas[i], self.u0s[i], self.lambd, self.rho)
                  for i in range(self.blocks)]
        pool = multiprocessing.Pool(self.processes)
        self.z0s = pool.map(mp.mp_z0_update, inputs)
        pool.close()
        #pool.join()

    def z1_z2_update(self):
        inputs = [(self.thetas[i], self.thetas[i-1],
                   self.u1s[i], self.u1s[i-1], self.u2s[i],
                   self.beta, self.rho)
                  for i in range(1, self.blocks)]
        pool = multiprocessing.Pool(self.processes)
        zs = pool.map(mp.mp_z1_z2_update, inputs)
        pool.close()
        #pool.join()
        for i in range(self.blocks - 1):
                self.z1s[i] = zs[i][0]
                self.z2s[i] = zs[i][1]

    def u_update(self):
        self.u0_update()
        self.u1_u2_update()

    def u0_update(self):
        inputs = [(self.thetas[i], self.u0s[i], self.z0s[i])
                  for i in range(self.blocks)]
        pool = multiprocessing.Pool(self.processes)
        self.u0s = pool.map(mp.mp_u0_update, inputs)
        pool.close()
        #pool.join()

    def u1_u2_update(self):
        inputs = [(self.thetas[i], self.thetas[i-1],
                   self.u1s[i-1], self.u2s[i],
                   self.z1s[i-1], self.z2s[i])
                  for i in range(1, self.blocks)]
        pool = multiprocessing.Pool(self.processes)
        us = pool.map(mp.mp_u1_u2_update, inputs)
        pool.close()
        #pool.join()
        for i in range(self.blocks - 1):
                self.u1s[i] = us[i][0]
                self.u2s[i] = us[i][1]
