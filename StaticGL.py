
from TVGL import TVGL
from DataHandler import DataHandler
import penalty_functions as pf
import numpy as np


class StaticGL(TVGL):

    def __init__(self, *args, **kwargs):
        super(StaticGL, self).__init__(blocks=1, beta=0, processes=1,
                                       *args, **kwargs)
        self.nju = float(self.obs)/float(self.rho)
        self.iteration = "n/a"
        self.penalty_function = "n/a"
        self.e = 1e-7

    def get_rho(self):
        return self.obs + 1

    def generate_real_thetas(self, line, splitter):
        dh = DataHandler()
        infos = line.split(splitter)
        for network_info in infos:
            filename = network_info.split(":")[0].strip("#").strip()
            dh.read_network(filename, inversion=False)
        self.real_thetas = dh.inverse_sigmas
        dh = None

    def theta_update(self):
        a = self.z0s[0] - self.u0s[0]
        at = a.transpose()
        m = self.nju*(a + at)/2 - self.emp_cov_mat[0]
        d, q = np.linalg.eig(m)
        qt = q.transpose()
        sqrt_matrix = np.sqrt(d**2 + 4/self.nju*np.ones(self.dimension))
        diagonal = np.diag(d) + np.diag(sqrt_matrix)
        self.thetas[0] = np.real(
            self.nju/2*np.dot(np.dot(q, diagonal), qt))

    def z_update(self):
        self.z0s[0] = pf.soft_threshold_odd(self.thetas[0] + self.u0s[0],
                                            self.lambd, self.rho)

    def u_update(self):
        self.u0s[0] = self.u0s[0] + self.thetas[0] - self.z0s[0]

    def temporal_deviations(self):
        self.deviations = ["n/a"]
        self.dev_ratio = "n/a"

    def correct_edges(self):
        self.real_edges = 0
        self.real_edgeless = 0
        self.correct_positives = 0
        self.all_positives = 0
        for real_network in self.real_thetas:
            for i in range(self.dimension - 1):
                for j in range(i + 1, self.dimension):
                    if real_network[i, j] != 0:
                        self.real_edges += 1
                        if self.thetas[0][i, j] != 0:
                            self.correct_positives += 1
                            self.all_positives += 1
                    elif real_network[i, j] == 0:
                        self.real_edgeless += 1
                        if self.thetas[0][i, j] != 0:
                            self.all_positives += 1
        self.precision = float(self.correct_positives)/float(
            self.all_positives)
        self.recall = float(self.correct_positives)/float(
            self.real_edges)
        self.f1score = 2*(self.precision*self.recall)/float(
            self.precision + self.recall)
