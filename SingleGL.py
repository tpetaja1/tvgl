
from BaseGraphicalLasso import BaseGraphicalLasso
from DataHandler import DataHandler
import penalty_functions as pf
import numpy as np
import time
import sys


class SingleGL(BaseGraphicalLasso):

    # Child class of BaseGraphicalLasso class.
    # computes a single Graphical Lasso problem
    # for the whole data set

    def __init__(self, *args, **kwargs):
        super(SingleGL, self).__init__(blocks=1, beta=0, processes=1,
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
        self.norm_deviations = ["n/a"]
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


if __name__ == "__main__" and len(sys.argv) == 3:

    # Input parameters from command line:
    #  1. Data file in csv format
    #  2. lambda

    start_time = time.time()
    datahandler = DataHandler()

    """ Parameters for creating solver instance """
    filename = sys.argv[1]
    real_data = True
    if "synthetic_data" in filename:
        real_data = False
    lambd = float(sys.argv[2])

    """ Create solver instance """
    print "\nReading file: %s\n" % filename
    solver = SingleGL(filename=filename,
                      lambd=lambd,
                      datecolumn=real_data)
    print "Total data samples: %s" % solver.datasamples
    print "Blocks: %s" % solver.blocks
    print "Observations in a block: %s" % solver.obs
    print "Rho: %s" % solver.rho
    print "Lambda: %s" % solver.lambd
    print "Beta: %s" % solver.beta
    print "Penalty function: %s" % solver.penalty_function
    print "Processes: %s" % solver.processes

    """ Run algorithm """
    print "\nRunning algorithm..."
    solver.run_algorithm()

    """ Evaluate and print results """
    print "\nNetwork 0:"
    for j in range(solver.dimension):
        print solver.thetas[0][j, :]
    print "\nTemporal deviations: "
    solver.temporal_deviations()
    print solver.deviations
    print "Normalized Temporal deviations: "
    print solver.norm_deviations
    try:
        print "Temp deviations ratio: {0:.3g}".format(solver.dev_ratio)
    except ValueError:
        print "Temp deviations ratio: n/a"

    """ Evaluate and create result file """
    if not real_data:
        solver.correct_edges()
        print "\nTotal Edges: %s" % solver.real_edges
        print "Correct Edges: %s" % solver.correct_positives
        print "Total Zeros: %s" % solver.real_edgeless
        false_edges = solver.all_positives - solver.correct_positives
        print "False Edges: %s" % false_edges
        print "F1 Score: %s" % solver.f1score
        datahandler.write_results(filename, solver)
    else:
        datahandler.write_network_results(filename, solver)

    """ Running times """
    print "\nAlgorithm run time: %s seconds" % (solver.run_time)
    print "Execution time: %s seconds" % (time.time() - start_time)
