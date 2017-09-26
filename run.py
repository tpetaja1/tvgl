
import time
import sys
from DataHandler import DataHandler
from SerialTVGL import SerialTVGL
from ParallelTVGL import ParallelTVGL
from DynamicGL import DynamicGL
from StaticGL import StaticGL


if __name__ == "__main__" and len(sys.argv) > 1:
    # Input parameters from command line:
    #  1. Data file in csv format
    #  2. Penalty function
    #     1 = "element_wise"
    #     2 = "group_lasso"
    #     3 = "perturbed_node
    #  3. Number of blocks to be created
    #  4. lambda
    #  5. beta
    #  6. number of processes to be used
    #  OPTIONAL:
    #  7. Algorithm type - Default = "parallel"
    #     "serial", "dynamic", "static" accepted

    start_time = time.time()
    datahandler = DataHandler()

    """ Parameters for creating algorithm instance """
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
    algo_type = "parallel"
    blocks = int(sys.argv[3])
    lambd = float(sys.argv[4])

    """ Create algorithm instance, ParallelTVGL is default """
    print "\nReading file: %s\n" % filename
    if "serial" in sys.argv:
        algo_type = "serial"
        algorithm = SerialTVGL(filename=filename,
                               penalty_function=penalty_function,
                               blocks=blocks,
                               lambd=lambd,
                               beta=int(sys.argv[5]),
                               datecolumn=real_data)
    elif "dynamic" in sys.argv:
        algo_type = "dynamic"
        algorithm = DynamicGL(filename=filename,
                              penalty_function=penalty_function,
                              blocks=blocks,
                              lambd=lambd,
                              processes=int(sys.argv[5]),
                              datecolumn=real_data)
    elif "static" in sys.argv:
        algo_type = "static"
        algorithm = StaticGL(filename=filename,
                             lambd=int(sys.argv[3]),
                             datecolumn=real_data)
    else:
        algorithm = ParallelTVGL(filename=filename,
                                 penalty_function=penalty_function,
                                 blocks=blocks,
                                 lambd=lambd,
                                 beta=int(sys.argv[5]),
                                 processes=int(sys.argv[6]),
                                 datecolumn=real_data)
    print "Total data samples: %s" % algorithm.datasamples
    print "Blocks: %s" % algorithm.blocks
    print "Observations in a block: %s" % algorithm.obs
    print "Rho: %s" % algorithm.rho
    print "Lambda: %s" % algorithm.lambd
    print "Beta: %s" % algorithm.beta
    print "Penalty function: %s" % algorithm.penalty_function
    print "Processes: %s" % algorithm.processes

    """ Run algorithm """
    print "\nRunning algorithm..."
    algorithm.run_algorithm()

    """ Evaluate and print results """
    print "\nNetwork 0:"
    for j in range(algorithm.dimension):
        print algorithm.thetas[0][j, :]
    print "\nTemporal deviations: "
    algorithm.temporal_deviations()
    print algorithm.deviations
    print "Normalized Temporal deviations: "
    print algorithm.norm_deviations
    try:
        print "Temp deviations ratio: {0:.3g}".format(algorithm.dev_ratio)
    except ValueError:
        print "Temp deviations ratio: n/a"

    """ Evaluate and create result file """
    if not real_data:
        algorithm.correct_edges()
        print "\nTotal Edges: %s" % algorithm.real_edges
        print "Correct Edges: %s" % algorithm.correct_positives
        print "Total Zeros: %s" % algorithm.real_edgeless
        false_edges = algorithm.all_positives - algorithm.correct_positives
        print "False Edges: %s" % false_edges
        print "F1 Score: %s" % algorithm.f1score
        datahandler.write_results(filename, algo_type, algorithm)
    else:
        datahandler.write_network_results(filename, algo_type, algorithm)

    """ Running times """
    print "\nAlgorithm run time: %s seconds" % (algorithm.run_time)
    print "Execution time: %s seconds" % (time.time() - start_time)
