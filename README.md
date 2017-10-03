# Time-Varying Graphical Lasso #

Contains 2 TVGL and 2 GL solvers for network inference,
and a DataHandler to create and maintain data files.

The solvers use an ADMM algorithm to solve the problem.

Implemented in Python2.7 with NumPy and multiprocessing.

Based on paper from Hallac et al. (2017)
  "Network inference via the Time-Varying Graphical Lasso"  
  https://arxiv.org/abs/1703.01958

## 1. Contains ##
   Files:
   - BaseGraphicalLasso.py -   Parent class for all 4 solvers
   - ParallelTVGL.py       -   TVGL solver using multiprocessing
   - SerialTVGL.py         -   TVGL solver using serial processing
   - StaticGL.py           -   GL solver solving T independent GL problems
   - SingleGL.py           -   GL solver for a single GL problem
   - DataHandler.py        -   Synthetic data generator, results writer
   - penalty_functions.py  -   Penalty functions defined for the algorithms

   Folders:
   - omxh_data             -   OMXH25 daily returns data
   - synthetic_data        -   Synthetic data generated with DataHandler
   - networks              -   Network files to be used for generating synthetic data
   - network_results       -   Results from non-synthetic data executions
   - results               -   Results from synthetic data executions
 		      
## 2. Run solvers ##

 Run from command line with following arguments:

 - Parallel TVGL
    1. [data file]
    2. [penalty function index]
    3. [Number of blocks to be created]
    4. [lambda]
    5. [beta]
    6. [number of processes to be used]  
       -if higher than # of blocks, # of processes is reduced to # of blocks

    Example:
       $ python ParallelTVGL.py synthetic_data/datafile1.csv 2 10 2 4 4

 - Serial TVGL
    1. [data file]
    2. [penalty function index]
    3. [Number of blocks to be created]
    4. [lambda]
    5. [beta]

    Example:  
       $ python SerialTVGL.py synthetic_data/datafile1.csv 2 10 2 4

 - Static GL
    1. [data file]
    2. [Number of blocks to be created]
    3. [lambda]
    4. [number of processes to be used]  
       -if higher than # of blocks, # of processes is reduced to # of blocks

    Example:
       $ python StaticGL.py synthetic_data/datafile1.csv 2 10 2

 - Single GL
    1. [data file]
    2. [lambda]

    Example:
       $ python SingleGL.py synthetic_data/datafile1.csv 2

## 3. Generate synthetic data ##

 Run from command line with at least one [network file name] [number of data points] pair as arguments.
 The first element of the pair defines the file from where network statistics are to be created.
 The second element of the pair defines the number of datapoints to be created from the given network.

 Given multiple pairs, the datapoints will be appended in the given order to the generated data file.

 Arbitrary number of pairs can be inputted.

 Example:  
     $ python DataHandler.py networks/network1.csv 200 networks/network2.csv 200 networks/network1.csv 200

