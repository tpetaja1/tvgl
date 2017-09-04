
from TVGL import TVGL
import penalty_functions as pf
import numpy as np
import multiprocessing
import time
import traceback

MAX_ITER = 5000


def mp_parallel_tvgl((thetas, z0s, z1s, z2s, u0s, u1s, u2s,
                      emp_cov_mat, lambd, beta, rho, nju,
                      indexes, out_queue, prev_pipe, next_pipe,
                      proc_index, stopping_criteria),
                     last=False):
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
        while iteration < MAX_ITER:  # and stopping_criteria is False:
            """ Theta Update """
            if next_pipe is not None:
                next_pipe.send((z2s[-1], u2s[-1]))
            if prev_pipe is not None:
                received = prev_pipe.recv()
                z2s[0], u2s[0] = received
            for j, i in zip(indexes[:end], range(nn)):
                a = (z0s[i] + z1s[i] + z2s[i] - u0s[i] - u1s[i] - u2s[i])/3
                at = a.transpose()
                m = nju*(a + at)/2 - emp_cov_mat[i]
                d, q = np.linalg.eig(m)
                qt = q.transpose()
                sqrt_matrix = np.sqrt(d**2 + 4/nju*np.ones(dimension))
                diagonal = np.diag(d) + np.diag(sqrt_matrix)
                thetas[i] = np.real(
                    nju/2*np.dot(np.dot(q, diagonal), qt))
                final_thetas[j] = thetas[i]
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
            for i in range(1, n):
                a = thetas[i] - thetas[i-1] + u2s[i] - u1s[i-1]
                e = pf.group_lasso_penalty(a, 2*beta/rho)
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
                if fro_norm < 1e-5:
                    stopping_criteria[proc_index] = True
            if all(criteria is True for criteria in stopping_criteria):
                if next_pipe is not None:
                    next_pipe.send(None)
                if prev_pipe is not None:
                    prev_pipe.send(None)
                break
            thetas_pre = list(thetas)
            iteration += 1
        out_queue.put((final_thetas, iteration))
    except Exception as e:
        traceback.print_exc()
        raise e


class ParallelTVGL(TVGL):

    def __init__(self, filename, blocks=10,
                 lambd=20, beta=20, processes=1):
        super(ParallelTVGL, self).__init__(filename,
                                           blocks, lambd,
                                           beta, processes)
        if self.processes > self.blocks:
            self.processes = self.blocks
        self.chunk = int(np.round(self.blocks/float(self.processes)))
        self.iteration = "n/a"

    def init_algorithm(self):
        self.results = multiprocessing.JoinableQueue()
        self.pipes = [multiprocessing.Pipe() for i in range(self.processes-1)]
        self.procs = []
        stopping_criteria = multiprocessing.Manager().list()
        for i in range(self.processes):
            stopping_criteria.append(False)
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
                                                   stopping_criteria),))
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
                                                   stopping_criteria), True))
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
                                                   stopping_criteria),))
            self.procs.append(p)

    def run_algorithm(self, max_iter=10000):
        self.init_algorithm()
        start_time = time.time()
        for p in self.procs:
            p.start()
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
        self.iteration = iteration
        print self.iteration
        self.run_time = '{0:.3g}'.format(time.time() - start_time)
        self.thetas = [np.round(theta, 3) for theta in self.thetas]
