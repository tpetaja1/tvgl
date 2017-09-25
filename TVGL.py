
import numpy as np
import time
from DataHandler import DataHandler


class TVGL(object):

    np.set_printoptions(precision=3)

    def __init__(self, filename, blocks, lambd, beta,
                 processes, penalty_function="group_lasso",
                 datecolumn=True):
        self.datecolumn = datecolumn
        self.processes = processes
        self.blocks = blocks
        self.penalty_function = penalty_function
        self.dimension = None
        self.emp_cov_mat = [0] * self.blocks
        self.real_thetas = [0] * self.blocks
        if self.datecolumn:
            self.blockdates = [0] * self.blocks
        self.read_data(filename)
        self.rho = self.get_rho()
        self.max_step = 0.1
        self.lambd = lambd
        self.beta = beta
        print "Rho: %s" % self.rho
        print "Lambda: %s" % self.lambd
        print "Beta: %s" % self.beta
        self.thetas = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.z0s = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.z1s = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.z2s = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.u0s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.u1s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.u2s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.nju = float(self.obs)/float(3*self.rho)
        self.e = 1e-5
        self.roundup = 1

    def read_data(self, filename, comment="#", splitter=","):
        with open(filename, "r") as f:
            comment_count = 0
            for i, line in enumerate(f):
                if comment in line:
                    comment_count += 1
                else:
                    if self.dimension is None:
                        if self.datecolumn:
                            self.dimension = len(line.split(splitter)) - 1
                        else:
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
                    if i == 1:
                        self.generate_real_thetas(line, splitter)
                    continue
                if count == 0 and self.datecolumn is True:
                    start_date = line.strip().split(splitter)[0]
                if self.datecolumn:
                    lst.append([float(x)
                                for x in np.array(line.strip().
                                                  split(splitter)[1:])])
                else:
                    lst.append([float(x)
                                for x in np.array(line.strip().
                                                  split(splitter))])
                count += 1
                if count == self.obs:
                    if self.datecolumn:
                        end_date = line.strip().split(splitter)[0]
                        self.blockdates[block] = start_date + " - " + end_date
                    datablck = np.array(lst)
                    tp = datablck.transpose()
                    self.emp_cov_mat[block] = np.real(
                        np.dot(tp, datablck)/self.obs)
                    lst = []
                    count = 0
                    block += 1

    def generate_real_thetas(self, line, splitter):
        dh = DataHandler()
        infos = line.split(splitter)
        for network_info in infos:
            filename = network_info.split(":")[0].strip("#").strip()
            datacount = network_info.split(":")[1].strip()
            sub_blocks = int(datacount)/self.obs
            for i in range(sub_blocks):
                dh.read_network(filename, inversion=False)
        self.real_thetas = dh.inverse_sigmas
        dh = None

    def get_rho(self):
        #if self.obs % 3 == 0:
        #    return self.obs / 3 + 1
        #else:
        #    return np.ceil(float(self.obs) / float(3))
        return float(self.obs + 0.1) / float(3)
        #return 2*self.obs

    def adjust_rho(self):
        step = 1/float(self.nju*2)
        step = max(self.max_step, step/1.1)
        self.rho = float(2*self.obs*step)/float(3)
        #self.rho = max(float(self.obs + 0.1) / float(3), self.rho/1.1)
        #if 3*self.rho - self.obs <= 0.01:
        #    return
        #self.rho = (np.exp(float(-self.iteration)/100) + self.obs)/3
        #if self.rho == self.max_rho:
        #    return
        #self.rho = min(self.rho * 1.1, self.max_rho)
        self.nju = float(self.obs)/float(3*self.rho)
        print "Rho: %s" % self.rho
        print "Nju: %s" % self.nju

    def run_algorithm(self, max_iter=10000):
        self.init_algorithm()
        self.iteration = 0
        stopping_criteria = False
        thetas_pre = []
        start_time = time.time()
        while self.iteration < max_iter and stopping_criteria is False:
            if self.iteration % 500 == 0 or self.iteration == 1:
                print "\n*** Iteration %s ***" % self.iteration
                print "Time passed: {0:.3g}s".format(time.time() - start_time)
                print "Rho: %s" % self.rho
                print "Nju: %s" % self.nju
                print "Step: {0:.3f}".format(1/(2*self.nju))
            if self.iteration % 500 == 0 or self.iteration == 1:
                s_time = time.time()
            self.theta_update()
            if self.iteration % 500 == 0 or self.iteration == 1:
                print "Theta update: {0:.3g}s".format(time.time() - s_time)
            if self.iteration % 500 == 0 or self.iteration == 1:
                s_time = time.time()
            self.z_update()
            if self.iteration % 500 == 0 or self.iteration == 1:
                print "Z-update: {0:.3g}s".format(time.time() - s_time)
            if self.iteration % 500 == 0 or self.iteration == 1:
                s_time = time.time()
            self.u_update()
            if self.iteration % 500 == 0 or self.iteration == 1:
                print "U-update: {0:.3g}s".format(time.time() - s_time)
            """ Check stopping criteria """
            if self.iteration % 500 == 0 or self.iteration == 1:
                s_time = time.time()
            if self.iteration > 0:
                fro_norm = 0
                for i in range(self.blocks):
                    dif = self.thetas[i] - thetas_pre[i]
                    fro_norm += np.linalg.norm(dif)
                if fro_norm < self.e:
                    stopping_criteria = True
            thetas_pre = list(self.thetas)
            self.iteration += 1
            #print "iter %s" % self.iteration
            #self.adjust_rho()
        self.run_time = "{0:.3g}".format(time.time() - start_time)
        self.final_tuning(stopping_criteria, max_iter)

    def theta_update(self):
        pass

    def z_update(self):
        pass

    def u_update(self):
        pass

    def terminate_pools(self):
        pass

    def init_algorithm(self):
        pass

    def final_tuning(self, stopping_criteria, max_iter):
        self.thetas = [np.round(theta, self.roundup) for theta in self.thetas]
        #self.only_true_false_edges()
        self.terminate_pools()
        if stopping_criteria:
            print "\nIterations to complete: %s" % self.iteration
        else:
            print "\nMax iterations (%s) reached" % max_iter

    def only_true_false_edges(self):
        for k in range(self.blocks):
            for i in range(self.dimension - 1):
                for j in range(i + 1, self.dimension):
                    if self.thetas[k][i, j] != 0:
                        self.thetas[k][i, j] = 1
                        self.thetas[k][j, i] = 1
                    else:
                        self.thetas[k][i, j] = 0
                        self.thetas[k][j, i] = 0

    def temporal_deviations(self):
        deviations = np.zeros(self.blocks - 1)
        for i in range(0, self.blocks - 1):
            dif = self.thetas[i+1] - self.thetas[i]
            np.fill_diagonal(dif, 0)
            deviations[i] = np.linalg.norm(dif)
        print deviations
        try:
            self.deviations = deviations/max(deviations)
            self.dev_ratio = float(max(deviations))/float(np.mean(deviations))
        except ZeroDivisionError:
            self.deviations = deviations
            self.dev_ratio = 0
        print "Temp deviations ratio: {0:.3g}".format(self.dev_ratio)

    def correct_edges(self):
        self.real_edges = 0
        self.real_edgeless = 0
        self.correct_positives = 0
        self.all_positives = 0
        for real_network, network in zip(self.real_thetas, self.thetas):
            for i in range(self.dimension - 1):
                for j in range(i + 1, self.dimension):
                    if real_network[i, j] != 0:
                        self.real_edges += 1
                        if network[i, j] != 0:
                            self.correct_positives += 1
                            self.all_positives += 1
                    elif real_network[i, j] == 0:
                        self.real_edgeless += 1
                        if network[i, j] != 0:
                            self.all_positives += 1
        self.precision = float(self.correct_positives)/float(
            self.all_positives)
        self.recall = float(self.correct_positives)/float(
            self.real_edges)
        self.f1score = 2*(self.precision*self.recall)/float(
            self.precision + self.recall)
