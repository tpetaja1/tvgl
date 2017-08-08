
import time
import sys
from DataHandler import DataHandler
from SerialTVGL import SerialTVGL
from AsyncProTVGL import AsyncProTVGL
from MultiProTVGL import MultiProTVGL


if __name__ == "__main__" and len(sys.argv) > 1:
    datahandler = DataHandler()
    start_time = time.time()
    algo_type = sys.argv[1]
    filename = sys.argv[2]
    print "Reading file: %s" % filename
    if algo_type == "serial":
        algorithm = SerialTVGL(filename, blocks=int(sys.argv[3]))
    elif algo_type == "async":
        algorithm = AsyncProTVGL(filename, blocks=int(sys.argv[3]),
                                 processes=2)
    elif algo_type == "multi":
        algorithm = MultiProTVGL(filename)
    else:
        raise Exception("Invalid algorithm name")
    print "Running algorithm..."
    algorithm.run_algorithm()
    print "Algorithm run time: %s seconds" % (algorithm.run_time)
    print algorithm.thetas[0]
    algorithm.temporal_deviations()
    print "Temp deviations: "
    print algorithm.deviations
    datahandler.write_results(filename, algo_type, algorithm)
    print "Execution time: %s seconds" % (time.time() - start_time)
