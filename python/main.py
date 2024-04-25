import cvxpy as cp                  # For optimization
import numpy as np                  # For matrix operations
from scipy.special import comb      # For binomnial coefficient
import mosek                        # Solver
import time                         # Time measurement
import sys, getopt                  # Command line arguments
from typing import NamedTuple       # C-like structure, but immutable (not changeable after creation)
import argparse                     # For command line arguments
from mosek.fusion import *          # For optimization
import old_functions

# ToDo:
# - Branch and bound algorithm
# - Think about branch and cut algorithm
# - Implement the heuristic of the paper of Anjos, Kennings and Vannelli

# Define the instance structure
class Instance(NamedTuple):
    pairs: list
    costs: list
    lengths: list
    cost_matrix: np.matrix

# Parse the command line arguments
def argument_parser(argv):
    inputfile = ''

    try:
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group()
        group.add_argument("-i", "--ifile", help="input file")
        group.add_argument("-g", "--generate", help="generate instance", type=int)
        
        args = parser.parse_args()

        if args.ifile:
            inputfile = "instance/"+args.ifile
        if args.generate:
            generate_instance(int(args.generate))
            inputfile = "instance/instance.txt"
    except getopt.GetoptError:
        print('main.py -i <inputfile> -o <outputfile> -g <instance size>')
        sys.exit(2)
    print('Input file is "', inputfile)

    return inputfile

# Generate a instance file with n elements
def generate_instance(n):
    with open("instance/instance.txt", 'w') as file:
        for i in range(1, n+1):
            for j in range(i+1, n+1):
                file.write("e {} {} {} \n".format(i, j, np.random.randint(1, 10)))
        for i in range(1, n+1):
            file.write("l {} \n".format(np.random.randint(1, 10)))

# Read in the instances from the file
def read_instance(file_path):
    with open(file_path, 'r') as file:
        dim = int(file.readline())
        lengths = [*map(int, file.readline().split(",")),]
        costs = []
        pairs = []
        for i in range(0, dim):
            line = file.readline().split(",")
            for j in range(0, dim):
                if i < j:
                    pairs.append((i+1,j+1))
                    costs.append(int(line[j]))
    
    cost_matrix = define_costmatrix(pairs, costs, lengths)

    return pairs, costs, lengths, cost_matrix

# Get the index of the element in the matrix
def get_index(i, j, n):
    return int(n*(i-1)-(((i-1)*(i-1)+(i-1))/2)+(j-i)-1)

# Create the cost matrix for the optimization problem
def define_costmatrix(pairs, costs, lengths):
    m = len(costs)
    n = len(lengths)
    cost_matrix = np.zeros((m,m))
    l = 1
    for (a,b) in pairs:
        for (c,d) in pairs[l:m]:
            if a == c:
                k, i, j = a, b, d
                x = get_index(k,i,n)
                y = get_index(k,j,n)
                cost = costs[get_index(i,j,n)]/2 * lengths[k-1]
                cost_matrix[x][y] = cost
            elif b == c:
                k, i, j = b, a, d
                x = get_index(i,k,n)
                y = get_index(k,j,n)
                cost = costs[get_index(i,j,n)]/2 * lengths[k-1]
                cost_matrix[x][y] = -cost
            elif b == d:
                k, i, j = b, a, c
                x = get_index(i,k,n)
                y = get_index(j,k,n)
                cost = costs[get_index(i,j,n)]/2 * lengths[k-1]
                cost_matrix[x][y] = cost
        l += 1

    return cost_matrix


    costsub = []
    costval = []
    for i in range(0, len(matrix)):
        helpersub = []
        helperval = []
        for j in range(i, len(matrix)):
            if matrix[i][j] != 0:
                helpersub.append([i,j])
                helperval.append(matrix[i][j])
        costsub.append(helpersub)
        costval.append(helperval)
    
    return costsub, costval

# Define the optimization problem with the mosek solver
def problem_mosek(instance, sum_triangle = False):
    pairs, costs, lengths = instance.pairs, instance.costs, instance.lengths    
    costs_matrix = define_costmatrix(pairs, costs, lengths)
    K = np.sum(costs)*np.sum(lengths)/2

    n = len(lengths)

    with Model("Test") as M:

        dim = len(pairs)
        X = M.variable("X", Domain.inPSDCone(dim))
        C = Matrix.sparse(costs_matrix) 

        # Objective function
        M.objective(ObjectiveSense.Minimize, Expr.sub(K, Expr.dot(C, X)))

        # Diagonal constraint
        M.constraint("diag",Expr.mulDiag(X, Matrix.eye(dim)), Domain.equalsTo(1.0))

        # Triangle constraints

        if(sum_triangle):
            for i,j in pairs:
                subi = []
                subj = []
                val = []
                for k in range(1, n+1):
                    ij, ik, jk = get_index(i,j,n), get_index(i,k,n), get_index(j,k,n)
                    if k == i or k == j:
                        continue
                    elif k < i:         # k < i < j -> ij * (-jk) = (-1) ||(-) ij * (-ik) = (+1) || (-) (-ik) * (-jk) = -1
                        subi.append(ij)
                        subj.append(jk)
                        val.append(-1.0)
                        subi.append(ij)
                        subj.append(ik)
                        val.append(1.0)
                        subi.append(ik)
                        subj.append(jk)
                        val.append(-1.0)
                    elif k < j:         # i < k < j -> ij * (-jk) = (-1) || (-) ij * (ik) = (-1) || (-) (ik) * (-jk) = +1
                        subi.append(ij)
                        subj.append(jk)
                        val.append(-1.0)
                        subi.append(ij)
                        subj.append(ik)
                        val.append(-1.0)
                        subi.append(ik)
                        subj.append(jk)
                        val.append(1.0)
                    else:               # i < j < k -> ij * jk = (+1) || (-) ij * ik = (-1) || (-) ik * jk = (-1)
                        subi.append(ij)
                        subj.append(jk)
                        val.append(1.0)
                        subi.append(ij)
                        subj.append(ik)
                        val.append(-1.0)
                        subi.append(ik)
                        subj.append(jk)
                        val.append(-1.0)

                #print(subi, subj, val)
                A = Matrix.sparse(dim, dim, subi, subj, val)   

                M.constraint(Expr.dot(A, X), Domain.equalsTo(2-n))         
        else:
            for i,j in pairs:
                for k in range(j+1, n+1):
                    ij, ik, jk = get_index(i,j,n), get_index(i,k,n), get_index(j,k,n)
                    M.constraint(Expr.sub(Expr.sub(X.index(ij,jk), X.index(ij,ik)), X.index(ik,jk)), Domain.equalsTo(-1.0))


        try:
            M.solve()
            
            M.acceptedSolutionStatus(AccSolutionStatus.Optimal)
            print("Optimal primal objective: {0}".format(M.primalObjValue()))

        except OptimizeError as e:
            print("Optimization failed. Error: {0}".format(e))

        except SolutionError as e:
            # The solution with at least the expected status was not available.
            # We try to diagnoze why.
            print("Requested solution was not available.")
            prosta = M.getProblemStatus()

            if prosta == ProblemStatus.DualInfeasible:
                print("Dual infeasibility certificate found.")

            elif prosta == ProblemStatus.PrimalInfeasible:
                print("Primal infeasibility certificate found.")
                
            elif prosta == ProblemStatus.Unknown:
            # The solutions status is unknown. The termination code
            # indicates why the optimizer terminated prematurely.
                print("The solution status is unknown.")
                symname, desc = mosek.Env.getcodedesc(mosek.rescode(int(M.getSolverIntInfo("optimizeResponse"))))
                print("   Termination code: {0} {1}".format(symname, desc))

            else:
                print("Another unexpected problem status {0} is obtained.".format(prosta))

        except Exception as e:
            print("Unexpected error: {0}".format(e))

# Transform the solution of the optimization problem to a permutation
def solution2permutation(X, n):
    R_values = []
    for i in X.value[:][0]:
        R_values.append(i)

    p_values = []
    for i in range(1, n+1):
        sum = 0
        for j in range(1, n+1):
            if i == j:
                continue
            elif i < j:
                sum += R_values[get_index(i,j,n)]
            else:
                sum -= R_values[get_index(j,i,n)]
        p_values.append(((sum+n+1)/2, i))
    p_values.sort(reverse=True)
    return p_values

# Get the length of facilities between i and j in the permutation
def getLength(p_values, lengths, i, j):
    length = (lengths[i-1] + lengths[j-1])/2
    found = False       # True if i have been found but j not found and false if i have not been found 
    for (value, index) in p_values:
        if (index == j or index == i) and found == True:
            break
        
        if found == True:
            length += lengths[index-1]
        
        if (index == i or index == j) and found == False:
            found = True
        
    return length

# Get the objective value of the permutation
def permutation2objective(p_values, costs, lengths, pairs):
    objective = 0
    for (x,y) in pairs:
        objective += costs[get_index(x,y,len(lengths))]*getLength(p_values, lengths, x, y)
    return objective

def main(argv):

    start_time = time.time()
    inputfile = argument_parser(argv)

    # Read the downloaded instance files from Hungerländer
    instance = Instance(*read_instance(inputfile))

    #problem, X = define_problem2(instance)

    problem_mosek(instance, False)  

    #p_values = solution2permutation(X, len(instance.lengths))

    #print(permutation2objective(p_values, instance.costs, instance.lengths, instance.pairs))

    #print(X.value)
    print("--- %s seconds ---" % (time.time() - start_time))

    #print(X.value())
    #print(M.primalObjValue())
    


if __name__ == "__main__":
    main(sys.argv[1:])