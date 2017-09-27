
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

    def write_network_results(self, datafile, alg_type, alg, splitter=","):
        run_time = datetime.datetime.now()
        results_name = "network_results/%s_di%sbl%sob%sla%sbe%s_%s.csv" % (
            alg_type, alg.dimension, alg.blocks, alg.obs, int(alg.lambd),
            int(alg.beta), run_time.strftime("%Y%m%d%H%M%S"))
        """ Read features """
        with open(datafile, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    feats = line.strip().split(splitter)[1:]
                    break
        features = {}
        for i, feat in enumerate(feats):
            features[i] = feat
        """ Write Results """
        with open(results_name, "w") as f:
            f.write("# Information\n")
            f.write("Run datetime, %s\n" %
                    run_time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write("Data file, %s\n" % datafile)
            f.write("Algorithm type, %s\n" % alg.__class__.__name__)
            f.write("Penalty function, %s\n" % alg.penalty_function)
            f.write("Data dimension, %s\n" % alg.dimension)
            f.write("Blocks, %s\n" % alg.blocks)
            f.write("Observations in a block, %s\n" % alg.obs)
            f.write("Rho, %s\n" % alg.rho)
            f.write("Beta, %s\n" % alg.beta)
            f.write("Lambda, %s\n" % alg.lambd)
            f.write("Processes used, %s\n" % alg.processes)
            f.write("\n")
            f.write("# Results\n")
            f.write("Algorithm run time, %s seconds\n" % alg.run_time)
            f.write("Iterations to complete, %s\n\n" % alg.iteration)
            try:
                f.write("Temporal deviations ratio (max/mean), {0:.3f}\n"
                        .format(alg.dev_ratio))
            except ValueError:
                f.write("Temporal deviations ratio (max/mean), %s\n"
                        % alg.dev_ratio)
            f.write("Temporal deviations ")
            for dev in alg.deviations:
                try:
                    f.write(",{0:.3f}".format(dev))
                except ValueError:
                    f.write(",%s" % dev)
            f.write("\nNormalized Temporal deviations ")
            for dev in alg.norm_deviations:
                try:
                    f.write(",{0:.3f}".format(dev))
                except ValueError:
                    f.write(",%s" % dev)
            """ Write networks """
            f.write("\n\n#Networks:\n\n")
            for k in range(alg.blocks):
                f.write("Block %s," % k)
                f.write(alg.blockdates[k] + "\n")
                if k > 0:
                    f.write("Dev to prev,")
                    f.write("{0:.3f},".format(alg.deviations[k-1]))
                if k < alg.blocks - 1:
                    f.write("Dev to next,")
                    f.write("{0:.3f}".format(alg.deviations[k]))
                f.write("\n")
                for feat in feats:
                    f.write("," + feat)
                f.write("\n")
                for i in range(alg.dimension):
                    f.write(features[i])
                    for j in range(alg.dimension):
                        f.write("," + str(alg.thetas[k][i, j]))
                    f.write("\n")
                f.write("\n\n")

    def write_results(self, datafile, alg_type, alg, splitter=','):
        run_time = datetime.datetime.now()
        results_name = "results/%s_di%sbl%sob%sla%sbe%s_%s.csv" % (
            alg_type, alg.dimension, alg.blocks, alg.obs, int(alg.lambd),
            int(alg.beta), run_time.strftime("%Y%m%d%H%M%S"))
        with open(results_name, "w") as f:
            f.write("# Information\n")
            f.write("Run datetime, %s\n" %
                    run_time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write("Data file, %s\n" % datafile)
            f.write("Algorithm type, %s\n" % alg.__class__.__name__)
            f.write("Penalty function, %s\n" % alg.penalty_function)
            f.write("Data dimension, %s\n" % alg.dimension)
            f.write("Blocks, %s\n" % alg.blocks)
            f.write("Observations in a block, %s\n" % alg.obs)
            f.write("Rho, %s\n" % alg.rho)
            f.write("Beta, %s\n" % alg.beta)
            f.write("Lambda, %s\n" % alg.lambd)
            f.write("Processes used, %s\n" % alg.processes)
            f.write("Total edges, %s\n" % alg.real_edges)
            f.write("Total edgeless, %s\n" % alg.real_edgeless)
            f.write("\n")
            f.write("# Results\n")
            f.write("Algorithm run time, %s seconds\n" % alg.run_time)
            f.write("Iterations to complete, %s\n\n" % alg.iteration)
            f.write("Correct positive edges, %s\n" % alg.correct_positives)
            f.write("All positives, %s\n" % alg.all_positives)
            f.write("F1 Score, {0:.3f}\n"
                    .format(alg.f1score))
            try:
                f.write("Temporal deviations ratio (max/mean), {0:.3f}\n"
                        .format(alg.dev_ratio))
            except ValueError:
                f.write("Temporal deviations ratio (max/mean), %s\n"
                        % alg.dev_ratio)
            f.write("Temporal deviations ")
            for dev in alg.deviations:
                try:
                    f.write(",{0:.3f}".format(dev))
                except ValueError:
                    f.write(",%s" % dev)
            f.write("\nNormalized Temporal deviations ")
            for dev in alg.norm_deviations:
                try:
                    f.write(",{0:.3f}".format(dev))
                except ValueError:
                    f.write(",%s" % dev)
            f.write("\n")


if __name__ == "__main__":
    dh = DataHandler()
    dh.read_network("networks/network1.csv")
    dh.read_network("networks/network3.csv")
    dh.generate_real_data([500, 500])
