
from TVGL import TVGL
import penalty_functions as pf
import numpy as np
import multiprocessing
import time
import traceback

MAX_ITER = 10000


def mp_parallel_tvgl((thetas, z0s, z1s, z2s, u0s, u1s, u2s,
                      emp_cov_mat, lambd, beta, rho, nju,
                      indexes, out_queue, prev_pipe, next_pipe,
                      proc_index, stopping_criteria, pen_func),
                     last=False):
    # Multiprocessing worker computing the TVGL algorithm for the given
    # set of blocks. Communicates the variables with neighboring
    # processes through multiprocessing Pipes, between every iteration.
    # Stopping criteria is fulfilled as every process reaches its own
    # stopping criteria. After global stopping criteria, poison pills
    # are sent to Pipes to terminate the iterations in every process.

    try:
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
        c = np.zeros((dimension, 3*dimension))
        c[:, 0:dimension] = np.eye(dimension)
        c[:, dimension:2*dimension] = -np.eye(dimension)
        c[:, 2*dimension:3*dimension] = np.eye(dimension)
        ct = c.transpose()
        cc = np.linalg.inv(np.dot(ct, c) + 2*np.eye(3*dimension))
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
                m = (a + at)/(2 * nju) - emp_cov_mat[i]
                d, q = np.linalg.eig(m)
                qt = q.transpose()
                sqrt_matrix = np.sqrt(d**2 + 4/nju*np.ones(dimension))
                diagonal = np.diag(d) + np.diag(sqrt_matrix)
                thetas[i] = np.real(
                    nju/2*np.dot(np.dot(q, diagonal), qt))
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


class ParallelTVGL(TVGL):

    def __init__(self, *args, **kwargs):
        super(ParallelTVGL, self).__init__(*args, **kwargs)
        if self.processes > self.blocks:
            self.processes = self.blocks
        self.chunk = int(np.round(self.blocks/float(self.processes)))

    def init_algorithm(self):

        """ Create result Queue, Pipes for communication between processes,
            initialize list for processes, stopping criteria Manager """
        self.results = multiprocessing.JoinableQueue()
        self.pipes = [multiprocessing.Pipe() for i in range(self.processes-1)]
        self.procs = []
        stopping_criteria = multiprocessing.Manager().list()
        for i in range(self.processes):
            stopping_criteria.append(False)

        """ Create processes. The blocks are divided into chunks based on their index.
            Each process will get a chunk of blocks,
            The last process gets the remaining blocks """
        for i in range(self.processes):
            if i == 0:
                p = multiprocessing.Process(target=mp_parallel_tvgl,
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
                                                   self.nju,
                                                   range(self.chunk * i, self.chunk*(i+1)+1),
                                                   self.results,
                                                   None,
                                                   self.pipes[i][0],
                                                   i,
                                                   stopping_criteria,
                                                   self.penalty_function),))
            elif i == self.processes - 1:
                p = multiprocessing.Process(target=mp_parallel_tvgl,
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
                                                   self.nju,
                                                   range(self.chunk * i, self.blocks),
                                                   self.results,
                                                   self.pipes[i-1][1],
                                                   None,
                                                   i,
                                                   stopping_criteria,
                                                   self.penalty_function), True))
            else:
                p = multiprocessing.Process(target=mp_parallel_tvgl,
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
                                                   self.nju,
                                                   range(self.chunk * i, self.chunk*(i+1)+1),
                                                   self.results,
                                                   self.pipes[i-1][1],
                                                   self.pipes[i][0],
                                                   i,
                                                   stopping_criteria,
                                                   self.penalty_function),))
            self.procs.append(p)

    def run_algorithm(self, max_iter=10000):

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
