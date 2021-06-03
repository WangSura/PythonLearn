import geatpy as ea # import geatpy
import numpy as np
from MyProblem2 import Ackley
if __name__ == '__main__':
    """=========================Instantiate your problem=========================="""
    problem = Ackley(30) # Instantiate MyProblem class.
    """===============================Population set=============================="""
    Encoding = 'RI'                # Encoding type.
    NIND =                       # Set the number of individuals.
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # Create the field descriptor.
    population = ea.Population(Encoding, Field, NIND) # Instantiate Population class(Just instantiate, not initialize the population yet.)
    """================================Algorithm set==============================="""
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population) # Instantiate a algorithm class.
    myAlgorithm.MAXGEN = 1000      # Set the max times of iteration.
    myAlgorithm.mutOper.F = 0.5    # Set the F of DE
    myAlgorithm.recOper.XOVR = 0.2 # Set the Cr of DE (Here it is marked as XOVR)
    myAlgorithm.logTras = 1        # Set the frequency of logging. If it is zero, it would not log.
    myAlgorithm.verbose = True     # Set if we want to print the log during the evolution or not.
    myAlgorithm.drawing = 1        # 1 means draw the figure of the result.
    """===============================Start evolution=============================="""
    [BestIndi, population] = myAlgorithm.run() # Run the algorithm templet.
    """==============================Output the result============================="""
    print('The number of evolution is: %s'%(myAlgorithm.evalsNum))
    if BestIndi.sizes != 0:
        print('The objective value of the best solution is: %s' % BestIndi.ObjV[0][0])
    else:
        print('Did not find any feasible solution.')