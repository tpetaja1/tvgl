
from BaseGraphicalLasso import BaseGraphicalLasso
from DataHandler import DataHandler
import penalty_functions as pf
import numpy as np
from multiprocessing import Process, Pipe, Manager, JoinableQueue
import time
import traceback
import sys

MAX_ITER = 10000


def mp_parallel_tvgl((thetas, z0s, z1s, z2s, u0s, u1s, u2s,
                      emp_cov_mat, lambd, beta, rho, eta,
                      indexes, out_queue, prev_pipe, next_pipe,
                      proc_index, stopping_criteria, pen_func),
                     last=False):

    # Multiprocessing worker computing the TVGL algorithm for the given
    # subset of blocks. Communicates the variables with neighboring
    # processes through multiprocessing Pipes, between every iteration.
    # Stopping criteria is fulfilled as every process reaches its own
    # stopping criteria. After global stopping criteria, poison pills
    # are sent to Pipes to terminate the iterations in every process.

    try:
        """ Define initial variables """
        iteration = 0
        n = len(indexes)
        if last:
            nn = n
            end = None
        else:
            nn = n - 1
            end = -1
        thetas_pre = []
        final_thetas = {}
        dimension = np.shape(thetas[0])[0]

        """ Compute offline the multiplication coefficients used in Z1Z2 update
            of perturbed node penalty """
        if pen_func == "perturbed_node":
            c = np.zeros((dimension, 3*dimension))
            c[:, 0:dimension] = np.eye(dimension)
            c[:, dimension:2*dimension] = -np.eye(dimension)
            c[:, 2*dimension:3*dimension] = np.eye(dimension)
            ct = c.transpose()
            cc = np.linalg.inv(np.dot(ct, c) + 2*np.eye(3*dimension))

        """ Run ADMM algorithm """
        while iteration < MAX_ITER:

            """ Send last Z2, U2 values to next process,
                Receive first Z2, U2 values from previous process """
            if next_pipe is not None:
                next_pipe.send((z2s[-1], u2s[-1]))
            if prev_pipe is not None:
                received = prev_pipe.recv()
                if received is None:
                    break
                z2s[0], u2s[0] = received

                """ Theta Update """
            for j, i in zip(indexes[:end], range(nn)):
                a = (z0s[i] + z1s[i] + z2s[i] - u0s[i] - u1s[i] - u2s[i])/3
                at = a.transpose()
                m = (a + at)/(2 * eta) - emp_cov_mat[i]
                d, q = np.linalg.eig(m)
                qt = q.transpose()
                sqrt_matrix = np.sqrt(d**2 + 4/eta*np.ones(dimension))
                diagonal = np.diag(d) + np.diag(sqrt_matrix)
                thetas[i] = np.real(
                    eta/2*np.dot(np.dot(q, diagonal), qt))
                final_thetas[j] = thetas[i]

            """ Send first Theta value to previous process,
                Receive last Theta value from next process """
            if prev_pipe is not None:
                prev_pipe.send(thetas[0])
            if next_pipe is not None:
                received = next_pipe.recv()
                if received is None:
                    break
                thetas[-1] = received

            """ Z0 Update """
            for i in range(nn):
                z0s[i] = pf.soft_threshold_odd(thetas[i] + u0s[i], lambd, rho)

            """ Z1-Z2 Update """
            if pen_func == "perturbed_node":
                for i in range(1, n):
                    z1s[i-1], z2s[i] = pf.perturbed_node(thetas[i-1],
                                                         thetas[i],
                                                         u1s[i-1],
                                                         u2s[i],
                                                         beta,
                                                         rho,
                                                         ct,
                                                         cc)
            else:
                for i in range(1, n):
                    a = thetas[i] - thetas[i-1] + u2s[i] - u1s[i-1]
                    e = getattr(pf, pen_func)(a, beta, rho)
                    summ = thetas[i] + thetas[i-1] + u2s[i] + u1s[i-1]
                    z1s[i-1] = 0.5*(summ - e)
                    z2s[i] = 0.5*(summ + e)

            """ U0 Update """
            for i in range(nn):
                u0s[i] = u0s[i] + thetas[i] - z0s[i]

            """ U1-U2 Update """
            for i in range(1, n):
                u1s[i-1] = u1s[i-1] + thetas[i-1] - z1s[i-1]
                u2s[i] = u2s[i] + thetas[i] - z2s[i]

            """ Check stopping criteria """
            if iteration > 0:
                fro_norm = 0
                for i in range(nn):
                    dif = thetas[i] - thetas_pre[i]
                    fro_norm += np.linalg.norm(dif)
                if fro_norm < 1e-4:
                    stopping_criteria[proc_index] = True
            if all(criteria is True for criteria in stopping_criteria):
                break
            thetas_pre = list(thetas)
            iteration += 1
            if iteration % 500 == 0 and proc_index == 0:
                print "*** Iteration: %s ***\n" % iteration

        """ When stopping criteria reached, send poison pills to pipes """
        if next_pipe is not None:
            next_pipe.send(None)
        if prev_pipe is not None:
            prev_pipe.send(None)

        """ Put final Thetas into result Queue """
        out_queue.put((final_thetas, iteration))
    except Exception as e:
        traceback.print_exc()
        raise e


class ParallelTVGL(BaseGraphicalLasso):

    # Child class of BaseGraphicalLasso class.
    # Computes TVGL problem in parallel

    def __init__(self, *args, **kwargs):
        super(ParallelTVGL, self).__init__(*args, **kwargs)
        if self.processes > self.blocks:
            self.processes = self.blocks
        self.chunk = int(np.round(self.blocks/float(self.processes)))

    def init_algorithm(self):

        """ Create result Queue, Pipes for communication between processes,
            initialize list for processes, stopping criteria Manager """
        self.results = JoinableQueue()
        self.pipes = [Pipe() for i in range(self.processes-1)]
        self.procs = []
        stopping_criteria = Manager().list()
        for i in range(self.processes):
            stopping_criteria.append(False)

        """ Create processes. The blocks are divided into chunks based on their index.
            Each process will get a chunk of blocks,
            The last process gets the remaining blocks """
        for i in range(self.processes):
            if i == 0:
                p = Process(target=mp_parallel_tvgl,
                            args=((self.thetas[self.chunk * i:self.chunk*(i+1)+1],
                                   self.z0s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.z1s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.z2s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.u0s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.u1s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.u2s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.emp_cov_mat[self.chunk * i:self.chunk*(i+1)+1],
                                   self.lambd,
                                   self.beta,
                                   self.rho,
                                   self.eta,
                                   range(self.chunk * i, self.chunk*(i+1)+1),
                                   self.results,
                                   None,
                                   self.pipes[i][0],
                                   i,
                                   stopping_criteria,
                                   self.penalty_function),))
            elif i == self.processes - 1:
                p = Process(target=mp_parallel_tvgl,
                            args=((self.thetas[self.chunk * i:],
                                   self.z0s[self.chunk * i:],
                                   self.z1s[self.chunk * i:],
                                   self.z2s[self.chunk * i:],
                                   self.u0s[self.chunk * i:],
                                   self.u1s[self.chunk * i:],
                                   self.u2s[self.chunk * i:],
                                   self.emp_cov_mat[self.chunk * i:],
                                   self.lambd,
                                   self.beta,
                                   self.rho,
                                   self.eta,
                                   range(self.chunk * i, self.blocks),
                                   self.results,
                                   self.pipes[i-1][1],
                                   None,
                                   i,
                                   stopping_criteria,
                                   self.penalty_function), True))
            else:
                p = Process(target=mp_parallel_tvgl,
                            args=((self.thetas[self.chunk * i:self.chunk*(i+1)+1],
                                   self.z0s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.z1s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.z2s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.u0s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.u1s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.u2s[self.chunk * i:self.chunk*(i+1)+1],
                                   self.emp_cov_mat[self.chunk * i:self.chunk*(i+1)+1],
                                   self.lambd,
                                   self.beta,
                                   self.rho,
                                   self.eta,
                                   range(self.chunk * i, self.chunk*(i+1)+1),
                                   self.results,
                                   self.pipes[i-1][1],
                                   self.pipes[i][0],
                                   i,
                                   stopping_criteria,
                                   self.penalty_function),))
            self.procs.append(p)

    def run_algorithm(self, max_iter=MAX_ITER):

        """ Initialize algorithm """
        self.init_algorithm()
        start_time = time.time()

        """ Start processes / algorithm """
        for p in self.procs:
            p.start()

        """ Get results """
        results = {}
        for i in range(self.processes):
            result, iteration = self.results.get()
            results.update(result)
            self.results.task_done()
        self.results.join()
        for i in results:
            self.thetas[i] = results[i]
        for p in self.procs:
            p.join()

        """ Perform final adjustments """
        self.iteration = iteration
        self.run_time = '{0:.3g}'.format(time.time() - start_time)
        self.final_tuning(True, MAX_ITER)

    def terminate_processes(self):
        for p in self.procs:
            p.terminate()


if __name__ == "__main__" and len(sys.argv) == 7:

    # Input parameters from command line:
    #  1. Data file in csv format
    #  2. Penalty function
    #     1 = "element_wise"
    #     2 = "group_lasso"
    #     3 = "perturbed_node"
    #  3. Number of blocks to be created
    #  4. lambda
    #  5. beta
    #  6. number of processes to be used

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
    processes = int(sys.argv[6])

    """ Create solver instance """
    print "\nReading file: %s\n" % filename
    solver = ParallelTVGL(filename=filename,
                          penalty_function=penalty_function,
                          blocks=blocks,
                          lambd=lambd,
                          beta=beta,
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
