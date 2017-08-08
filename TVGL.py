
import numpy as np
import time


class TVGL(object):

    np.set_printoptions(precision=3)

    def __init__(self, filename, blocks, processes):
        self.processes = processes
        self.blocks = blocks
        self.dimension = None
        self.emp_cov_mat = [0] * self.blocks
        self.read_data(filename)
        self.rho = 50
        self.lambd = 90
        self.beta = 4
        self.thetas = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.z0s = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.z1s = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.z2s = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.u0s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.u1s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.u2s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.nju = float(self.obs)/float(3*self.rho)

    def read_data(self, filename, comment="#", splitter=","):
        with open(filename, "r") as f:
            comment_count = 0
            for i, line in enumerate(f):
                if comment in line:
                    comment_count += 1
                else:
                    if self.dimension is None:
                        self.dimension = len(line.split(splitter))
        datasamples = i + 1 - comment_count
        print "Total data samples: %s" % datasamples
        self.obs = datasamples / self.blocks
        print "Blocks: %s" % self.blocks
        print "Observations in a block: %s" % self.obs
        with open(filename, "r") as f:
            lst = []
            block = 0
            count = 0
            for i, line in enumerate(f):
                if comment in line:
                    continue
                lst.append([float(x)
                            for x in np.array(line.strip().split(splitter))])
                count += 1
                if count == self.obs:
                    datablck = np.array(lst)
                    tp = datablck.transpose()
                    self.emp_cov_mat[block] = np.real(
                        np.dot(tp, datablck)/self.obs)
                    lst = []
                    count = 0
                    block += 1

    def run_algorithm(self, max_iter=2000):
        self.iteration = 0
        stopping_criteria = False
        thetas_pre = []
        start_time = time.time()
        while self.iteration < max_iter and stopping_criteria is False:
            self.theta_update()
            self.z_update()
            self.u_update()
            """ Check stopping criteria """
            if self.iteration > 0:
                fro_norm = 0
                for i in range(self.blocks):
                    dif = self.thetas[i] - thetas_pre[i]
                    fro_norm += np.linalg.norm(dif)
                if fro_norm < 1e-5:
                    stopping_criteria = True
            thetas_pre = list(self.thetas)
            self.iteration += 1
        self.run_time = '{0:.3g}'.format(time.time() - start_time)
        self.final_tuning(stopping_criteria, max_iter)

    def theta_update(self):
        pass

    def z_update(self):
        pass

    def u_update(self):
        pass

    def final_tuning(self, stopping_criteria, max_iter):
        self.thetas = [np.round(theta, 3) for theta in self.thetas]
        if stopping_criteria:
            print "Iterations to complete: %s" % self.iteration
        else:
            print "Max iterations (%s) reached" % max_iter

    def temporal_deviations(self):
        deviations = np.zeros(self.blocks - 1)
        for i in range(0, self.blocks - 1):
            dif = self.thetas[i+1] - self.thetas[i]
            deviations[i] = np.linalg.norm(dif)
        print deviations
        self.deviations = deviations/max(deviations)
