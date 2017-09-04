
import time
import sys
from DataHandler import DataHandler
from SerialTVGL import SerialTVGL
from AsyncProTVGL import AsyncProTVGL
from MultiProTVGL import MultiProTVGL
from ProcTVGL import ProcTVGL
from LastTVGL import LastTVGL
from ParallelTVGL import ParallelTVGL
from StaticGL import StaticGL


if __name__ == "__main__" and len(sys.argv) > 1:
    datahandler = DataHandler()
    start_time = time.time()
    algo_type = sys.argv[1]
    filename = sys.argv[2]
    blocks = int(sys.argv[3])
    lambd = int(sys.argv[4])
    beta = int(sys.argv[5])
    print "Reading file: %s" % filename
    if algo_type == "serial":
        algorithm = SerialTVGL(filename, blocks,
                               lambd, beta, 1)
    elif algo_type == "async":
        algorithm = AsyncProTVGL(filename, blocks,
                                 lambd, beta, processes=2)
    elif algo_type == "multi":
        algorithm = MultiProTVGL(filename, blocks,
                                 lambd, beta, processes=int(sys.argv[6]))
    elif algo_type == "proc":
        algorithm = ProcTVGL(filename, blocks,
                             lambd, beta, processes=int(sys.argv[6]))
    elif algo_type == "last":
        algorithm = LastTVGL(filename, blocks,
                             lambd, beta, processes=int(sys.argv[6]))
    elif algo_type == "parallel":
        algorithm = ParallelTVGL(filename, blocks,
                                 lambd, beta, processes=int(sys.argv[6]))
    elif algo_type == "static":
        algorithm = StaticGL(filename, blocks,
                             lambd, processes=int(sys.argv[5]))
    else:
        raise Exception("Invalid algorithm name")
    print "Running algorithm..."
    algorithm.run_algorithm()
    print "Algorithm run time: %s seconds" % (algorithm.run_time)
    print algorithm.thetas[0]
    print algorithm.thetas[-1]
    algorithm.temporal_deviations()
    print "Temp deviations: "
    print algorithm.deviations
    algorithm.correct_nonzero_elements()
    print "Total nonzeros: %s" % algorithm.real_nonzeros
    print "Nonzero ratio: %s" % algorithm.nonzero_ratio
    datahandler.write_results(filename, algo_type, algorithm)
    print "Execution time: %s seconds" % (time.time() - start_time)
