
from BaseGraphicalLasso import BaseGraphicalLasso
from DataHandler import DataHandler
import penalty_functions as pf
import numpy as np
import multiprocessing
import time
import traceback
import sys


def mp_static_gl((theta, z0, u0, emp_cov_mat, rho,
                  lambd, eta, dimension, max_iter)):

    # Multiprocessing worker computing the
    # Static Graphical Lasso for given subset
    # of blocks

    try:
        iteration = 0
        stopping_criteria = False
        theta_pre = []
        while iteration < max_iter and stopping_criteria is False:
            """ Theta update """
            a = z0 - u0
            at = a.transpose()
            m = eta*(a + at)/2 - emp_cov_mat
            d, q = np.linalg.eig(m)
            qt = q.transpose()
            sqrt_matrix = np.sqrt(d**2 + 4/eta*np.ones(dimension))
            diagonal = np.diag(d) + np.diag(sqrt_matrix)
            theta = np.real(
                eta/2*np.dot(np.dot(q, diagonal), qt))
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


class StaticGL(BaseGraphicalLasso):

    # Child class of BaseGraphicalLasso class.
    # Computes Static Graphical Lasso problem
    # in parallel.

    def __init__(self, *args, **kwargs):
        super(StaticGL, self).__init__(beta=0, *args, **kwargs)
        self.eta = float(self.obs)/float(self.rho)
        self.iteration = "n/a"
        self.penalty_function = "n/a"

    def get_rho(self):
        return self.obs + 1

    def run_algorithm(self, max_iter=10000):
        start_time = time.time()
        p = multiprocessing.Pool(self.processes)
        inputs = [(self.thetas[i], self.z0s[i], self.u0s[i],
                   self.emp_cov_mat[i], self.rho,
                   self.lambd, self.eta, self.dimension, max_iter)
                  for i in range(self.blocks)]
        self.thetas = p.map(mp_static_gl, inputs)
        p.close()
        p.join()
        self.run_time = '{0:.3g}'.format(time.time() - start_time)
        self.thetas = [np.round(theta, self.roundup) for theta in self.thetas]


if __name__ == "__main__" and len(sys.argv) == 5:

    # Input parameters from command line:
    #  1. Data file in csv format
    #  2. Number of blocks to be created
    #  3. lambda
    #  4. number of processes

    start_time = time.time()
    datahandler = DataHandler()

    """ Parameters for creating solver instance """
    filename = sys.argv[1]
    real_data = True
    if "synthetic_data" in filename:
        real_data = False
    blocks = int(sys.argv[2])
    lambd = float(sys.argv[3])
    processes = int(sys.argv[4])

    """ Create solver instance """
    print "\nReading file: %s\n" % filename
    solver = StaticGL(filename=filename,
                      blocks=blocks,
                      lambd=lambd,
                      processes=processes,
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
