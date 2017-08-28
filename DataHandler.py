
import numpy as np
import datetime


class DataHandler(object):

    def __init__(self):
        self.inverse_sigmas = []
        self.sigmas = []
        self.network_files = []

    def read_network(self, filename, comment="#", splitter=",",
                     inversion=True):
        nodes = []
        self.network_files.append(filename)
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if comment in line:
                    continue
                data = line.split(splitter)
                if data[0] not in nodes:
                    nodes.append(int(data[0]))
                if data[1] not in nodes:
                    nodes.append(int(data[1]))
        self.dimension = max(nodes)
        network = np.eye(self.dimension)
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if comment in line:
                    continue
                data = line.split(splitter)
                network[int(data[0])-1, int(data[1])-1] = float(data[2])
                network[int(data[1])-1, int(data[0])-1] = float(data[2])
        self.inverse_sigmas.append(network)
        if inversion:
            sigma = np.linalg.inv(network)
            print np.linalg.eigvals(sigma)
            self.sigmas.append(sigma)
            print sigma
            print np.shape(sigma)
            print network

    def init_true_inverse_covariance_matrices(self):
        inverse_sigma1 = np.array([[1.00, 0.50, 0.00, 0.00, 0.00, 0.00],
                                   [0.50, 1.00, 0.50, 0.25, 0.00, 0.00],
                                   [0.00, 0.50, 1.00, 0.00, 0.25, 0.00],
                                   [0.00, 0.25, 0.00, 1.00, 0.50, 0.00],
                                   [0.00, 0.00, 0.25, 0.50, 1.00, 0.25],
                                   [0.00, 0.00, 0.00, 0.00, 0.25, 1.00]])

        inverse_sigma2 = np.array([[1.00, 0.00, 0.00, 0.50, 0.00, 0.00],
                                   [0.00, 1.00, 0.00, 0.00, 0.50, 0.00],
                                   [0.00, 0.00, 1.00, 0.50, 0.25, 0.50],
                                   [0.50, 0.00, 0.50, 1.00, 0.00, 0.00],
                                   [0.00, 0.50, 0.25, 0.00, 1.00, 0.00],
                                   [0.00, 0.00, 0.50, 0.00, 0.00, 1.00]])
        self.inverse_sigmas.append(inverse_sigma1)
        self.inverse_sigmas.append(inverse_sigma2)
        self.sigmas.append(np.linalg.inv(inverse_sigma1))
        self.sigmas.append(np.linalg.inv(inverse_sigma2))

    def generate_real_data(self, counts=[100, 100]):
        if len(counts) is not len(self.sigmas):
            raise Exception(
                "Lengths of networks and data lengths do not match.")
        z = None
        total_count = 0
        for sigma, datacount in zip(self.sigmas, counts):
            x = np.random.multivariate_normal(np.zeros(self.dimension),
                                              sigma, datacount)
            total_count += datacount
            if z is None:
                z = x
            else:
                z = np.vstack((z, x))
        filename = "synthetic_data/%sx%s_%s.csv" % (
            total_count, self.dimension,
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        header = "# Data generated from networks:\n# "
        for f, datacount in zip(self.network_files, counts):
            header += "%s: %s, " % (f, datacount)
        header = header[:-2]
        header += "\n"
        with open(filename, "w") as new_file:
            new_file.write(header)
            for datarow in z:
                line = ""
                for value in datarow:
                    line += "," + str("{0:.4f}".format(value))
                line = line[1:]
                new_file.write("%s\n" % line)

    def write_results(self, datafile, alg_type, alg):
        run_time = datetime.datetime.now()
        results_name = "results/%s_d%sb%so%s_%s.txt" % (
            alg_type, alg.dimension, alg.blocks, alg.obs,
            run_time.strftime("%Y%m%d%H%M%S"))
        with open(results_name, "w") as f:
            f.write("# Information\n")
            f.write("Run datetime: %s\n" %
                    run_time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write("Data file: %s\n" % datafile)
            f.write("Algorithm type: %s\n" % alg_type)
            f.write("Data dimension: %s\n" % alg.dimension)
            f.write("Blocks: %s\n" % alg.blocks)
            f.write("Observations in a block: %s\n" % alg.obs)
            f.write("Rho: %s\n" % alg.rho)
            f.write("Beta: %s\n" % alg.beta)
            f.write("Lambda: %s\n" % alg.lambd)
            f.write("Processes used: %s\n\n" % alg.processes)
            f.write("# Results\n")
            f.write("Algorithm run time: %s seconds\n" % alg.run_time)
            f.write("Iterations to complete: %s\n\n" % alg.iteration)
            f.write("Matching edge ratio (nonzeros): {0:.3f}\n"
                    .format(alg.nonzero_ratio))
            f.write("Temporal deviations ratio (max/mean): {0:.3f}\n"
                    .format(alg.dev_ratio))
            f.write("Temporal deviations: ")
            for dev in alg.deviations:
                f.write("{0:.3f} ".format(dev))
            f.write("\n")


if __name__ == "__main__":
    dh = DataHandler()
    dh.read_network("networks/network1.csv")
    dh.read_network("networks/network2.csv")
    dh.generate_real_data([50, 50])
