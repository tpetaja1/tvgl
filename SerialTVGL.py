
from BaseGraphicalLasso import BaseGraphicalLasso
from DataHandler import DataHandler
import penalty_functions as pf
import numpy as np
import time
import sys


class SerialTVGL(BaseGraphicalLasso):

    # Child class of BaseGraphicalLasso class.
    # Computes TVGL problem in serial,
    # no parallelization

    def __init__(self, *args, **kwargs):
        super(SerialTVGL, self).__init__(processes=1, *args, **kwargs)

    def theta_update(self):
        for i in range(self.blocks):
            a = (self.z0s[i] + self.z1s[i] + self.z2s[i] -
                 self.u0s[i] - self.u1s[i] - self.u2s[i])/3
            at = a.transpose()
            m = self.eta*(a + at)/2 - self.emp_cov_mat[i]
            d, q = np.linalg.eig(m)
            qt = q.transpose()
            sqrt_matrix = np.sqrt(d**2 + 4/self.eta*np.ones(self.dimension))
            diagonal = np.diag(d) + np.diag(sqrt_matrix)
            self.thetas[i] = np.real(
                self.eta/2*np.dot(np.dot(q, diagonal), qt))

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
            ee = [getattr(pf, self.penalty_function)(a, self.beta, self.rho)
                  for a in aa]
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


if __name__ == "__main__" and len(sys.argv) == 6:

    # Input parameters from command line:
    #  1. Data file in csv format
    #  2. Penalty function
    #     1 = "element_wise"
    #     2 = "group_lasso"
    #     3 = "perturbed_node"
    #  3. Number of blocks to be created
    #  4. lambda
    #  5. beta

    start_time = time.time()
    datahandler = DataHandler()

    """ Parameters for creating solver instance """
    filename = sys.argv[1]
    real_data = True
    if "synthetic_data" in filename:
        real_data = False
    if sys.argv[2] == "1":
        penalty_function = "element_wise"
    elif sys.argv[2] == "2":
        penalty_function = "group_lasso"
    elif sys.argv[2] == "3":
        penalty_function = "perturbed_node"
    blocks = int(sys.argv[3])
    lambd = float(sys.argv[4])
    beta = float(sys.argv[5])

    """ Create solver instance """
    print "\nReading file: %s\n" % filename
    solver = SerialTVGL(filename=filename,
                        penalty_function=penalty_function,
                        blocks=blocks,
                        lambd=lambd,
                        beta=beta,
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
