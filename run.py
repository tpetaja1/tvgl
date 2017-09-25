
import time
import sys
from DataHandler import DataHandler
from SerialTVGL import SerialTVGL
from ParallelTVGL import ParallelTVGL
from DynamicGL import DynamicGL
from StaticGL import StaticGL


if __name__ == "__main__" and len(sys.argv) > 1:
    real_data = True
    datahandler = DataHandler()
    start_time = time.time()
    algo_type = sys.argv[1]
    filename = sys.argv[2]
    if "synthetic_data" in filename:
        real_data = False
    print "Reading file: %s" % filename
    if algo_type == "serial":
        algorithm = SerialTVGL(filename=filename,
                               blocks=int(sys.argv[3]),
                               lambd=int(sys.argv[4]),
                               beta=int(sys.argv[5]),
                               datecolumn=real_data)
    elif algo_type == "parallel":
        algorithm = ParallelTVGL(filename=filename,
                                 blocks=int(sys.argv[3]),
                                 lambd=int(sys.argv[4]),
                                 beta=int(sys.argv[5]),
                                 processes=int(sys.argv[6]),
                                 datecolumn=real_data,
                                 penalty_function="group_lasso")
    elif algo_type == "dynamic":
        algorithm = DynamicGL(filename=filename,
                              blocks=int(sys.argv[3]),
                              lambd=int(sys.argv[4]),
                              processes=int(sys.argv[5]),
                              datecolumn=real_data)
    elif algo_type == "static":
        algorithm = StaticGL(filename=filename,
                             lambd=int(sys.argv[3]),
                             datecolumn=real_data)
    else:
        raise Exception("Invalid algorithm name")
    print "Running algorithm..."
    algorithm.run_algorithm()
    print algorithm.thetas[0]
    #print algorithm.thetas[-1]
    algorithm.temporal_deviations()
    print "Temp deviations: "
    print algorithm.deviations
    #print "Emp cov matrices:"
    #print algorithm.emp_cov_mat[0]
    if not real_data:
        algorithm.correct_edges()
        print "Total Edges: %s" % algorithm.real_edges
        print "Correct Edges: %s" % algorithm.correct_positives
        print "Total Zeros: %s" % algorithm.real_edgeless
        false_edges = algorithm.all_positives - algorithm.correct_positives
        print "False Edges: %s" % false_edges
        print "F1 Score: %s" % algorithm.f1score
        datahandler.write_results(filename, algo_type, algorithm)
    else:
        datahandler.write_network_results(filename, algo_type, algorithm)
    print "Algorithm run time: %s seconds" % (algorithm.run_time)
    print "Execution time: %s seconds" % (time.time() - start_time)
