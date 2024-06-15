#import cvxpy as cp                  # For optimization
#import glob
import numpy as np                  # For matrix operations
from scipy.special import comb      # For binomnial coefficient
#import mosek                        # Solver
import time                         # Time measurement
import sys, getopt                  # Command line arguments
from typing import NamedTuple       # C-like structure, but immutable (not changeable after creation)
import argparse                     # For command line arguments
from mosek.fusion import *                 # For optimization
#import os
import matplotlib.pyplot as plt
import math
import mosek.fusion.pythonic
import random

EPSILON = 1
MAX_INT = 999999999
EPSOPT = 1e-4
 
class Logger:
 
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')
 
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
 
    def flush(self):
        self.console.flush()
        self.file.flush()
 
class Node():
    def __init__(self, index, level, lb, ub, constraints, Y, solved):
        self.index = index
        self.level = level
        self.lb = lb
        self.ub = ub
        self.constraints = constraints
        self.solved = solved
        self.Y = Y
    
    def __str__(self):
        return f"Node {self.index}, level {self.level}, lb {self.lb}, ub {self.ub}, constraints {self.constraints}"

    def __lt__(self, other):
        return self.lb < other.lb

# TODO change to class ignore NamedTuple
class Solution(NamedTuple):
    feasible: bool
    index: int
    level: int
    lb: float
    obj1: float
    obj2: float
    permutation: list
    Y: list
    
# NamedTuple for the instance as it is immutable
class Instance(NamedTuple):
    pairs: list
    lengths: list
    costs1: list
    costs2: list
    cost_matrix1: np.matrix
    cost_matrix2: np.matrix
    dim: int
    n: int
    K: float
    K2: float
    
# Parse the command line arguments
def argument_parser(argv):
    inputfile1 = ''
    inputfile2 = ''

    try:
        parser = argparse.ArgumentParser()
        #group = parser.add_mutually_exclusive_group()
        parser.add_argument("-i", "--ifile", help="input file1")
        parser.add_argument("-j", "--jfile", help="input file2")
        #group.add_argument("-g", "--generate", help="generate instance", type=int)
        
        args = parser.parse_args()

        if args.ifile:
            inputfile = "instance/"+args.ifile
        if args.jfile:
            inputfile2 = "instance/"+args.jfile
        #if args.generate:
            #generate_instance(int(args.generate))
            #inputfile = "instance/instance.txt"
    except getopt.GetoptError:
        print('main.py -i <inputfile> -j <inputfile> -o <outputfile>')
        sys.exit(2)

    return inputfile, inputfile2

# Generate a instance file with n elements
def generate_instance(n):
    with open("instance/instance.txt", 'w') as file:
        for i in range(1, n+1):
            for j in range(i+1, n+1):
                file.write("e {} {} {} \n".format(i, j, np.random.randint(1, 10)))
        for i in range(1, n+1):
            file.write("l {} \n".format(np.random.randint(1, 10)))

# Read in the instances from the file
def read_instance(instance_file1, instance_file2):
    with open(instance_file1, 'r') as file:
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
    K = np.sum(costs)*np.sum(lengths)/2
    
    # Code for the second objective function / not working with existing instances
    """
    if os.path.exists("cost_vector_"+str(len(costs))+".txt"):
        costs2 = read_cost_vector("cost_vector_"+str(len(costs))+".txt")
        cost_matrix2 = define_costmatrix(pairs, costs2, lengths)
    else:
        print("Cost vector does not exist")
        generate_cost_vector(len(costs))
        costs2 = read_cost_vector("cost_vector_"+str(len(costs))+".txt")
        cost_matrix2 = define_costmatrix(pairs, costs2, lengths)

    K2 = np.sum(costs2)*np.sum(lengths)/2"""
    
    costs2 = []
    with open(instance_file2, 'r') as file:
        file.readline()
        file.readline()
        costs2 = []
        for i in range(0, dim):
            line = file.readline().split(",")
            for j in range(0, dim):
                if i < j:
                    costs2.append(int(line[j]))
    
    cost_matrix2 = define_costmatrix(pairs, costs2, lengths)
    K2 = np.sum(costs2)*np.sum(lengths)/2
    
    return Instance(pairs, lengths, costs, costs2, cost_matrix, cost_matrix2, len(pairs), len(lengths), K, K2)

# Get the index of the element in the matrix
def get_index(i, j, n):
    return int(n*(i-1)-(((i-1)*(i-1)+(i-1))/2)+(j-i)-1)

# Create the cost matrix for the optimization problem
def define_costmatrix(pairs, costs, lengths):   # Try to build it as sparse matrix, as it has a lot of 0s
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

# Print the solution of the optimization problem / still too much other stuff in there
def print_solution(instance, M):

    pairs, costs, lengths = instance.pairs, instance.costs, instance.lengths

    dim = len(pairs)
    n = len(lengths)

    Z = M.getVariable("Z")
    Y = Z.slice([1,1] , [dim+1, dim+1])
    y = Z.slice([1,0], [1,dim+1])
            
    print("Root: ")
    M.acceptedSolutionStatus(AccSolutionStatus.Optimal)
    print("Relaxation objective value: {0}".format(M.primalObjValue()))
    
    rank = getRank(Y, dim)
    
    print("Rank of Y:", rank)

    p_values = solution2permutation(Y.level()[0:dim], n)
    print("The permutation of this solution is:", [x[1] for x in p_values])
    print("The objective value of this solution is:", permutation2objective(p_values, instance))
    
# Transform the solution of the optimization problem to a permutation
def solution2permutation(X, n, switch, pairs):
    # R_values = []
    # for i in X:
    #     R_values.append(i-EPSOPT)

    # R_values = np.array(X)

    # p_values = []
    # for i in range(1, n+1):
    #     sum = 0
    #     for j in range(1, n+1):
    #         if i == j:
    #             continue
    #         elif i < j:
    #             sum += R_values[get_index(i,j,n)]*switch
    #         else:
    #             sum -= R_values[get_index(j,i,n)]*switch
    #     p_values.append(((sum+n+1)/2, i))
    # #p_values.sort(reverse=True)

    R_values = np.zeros((n,n-1))    
    for i in range(len(pairs)):
        x=pairs[i][1]-2
        y=pairs[i][0]-1
        R_values[y,x] = X[i]
        R_values[x+1,y] = -X[i]
    
    p_help = R_values.sum(axis = 1)
    p_values = list(zip(p_help, range(1, n+1)))
    p_values.sort(reverse=True)
    # print("P_help", p_values)
                
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
def permutation2objective(p_values, instance):
    pairs, costs, costs2, lengths = instance.pairs, instance.costs, instance.costs2, instance.lengths
    objective = 0
    objective2 = 0
    for (x,y) in pairs:
        objective += costs[get_index(x,y,len(lengths))]*getLength(p_values, lengths, x, y)
        objective2 += costs2[get_index(x,y,len(lengths))]*getLength(p_values, lengths, x, y)
    return objective

def getRank(Y, dim):
    d = [0.0]*dim
    mosek.LinAlg.syeig(mosek.uplo.lo, dim, Y.level(), d)
    return sum([d[i] > 1e-6 for i in range(dim)])

def multi_obj_model(instance, M, vl, lb):
    pairs = instance.pairs
    costs_matrix = instance.cost_matrix1
    cost_matrix_2 = instance.cost_matrix2
    K = instance.K
    K2 = instance.K2

    n = instance.n
    dim =  instance.dim

    Z = M.variable("Z", Domain.inPSDCone(dim+1))
    Y = Z.slice([1,1] , [dim+1, dim+1])

    C = Matrix.sparse(costs_matrix)
    C2 = Matrix.sparse(cost_matrix_2)

    # Objective function
    obj1 = Expr.sub(K, Expr.dot(C, Y))
    obj2 = Expr.sub(K2, Expr.dot(C2, Y))
    M.objective(ObjectiveSense.Minimize, obj1)

    # Second objective constraint / Epsilon constraint
    M.constraint(obj2, Domain.lessThan(vl-EPSILON))
    
    # Diagonal equals to 1 constraints
    M.constraint("diag",Z.diag(), Domain.equalsTo(np.ones(dim+1))) # dim+1 constraints -> n over 2 + 1 constraints

    ## Betweeness constraints / 3-cycle constraints with vectorization -> way faster -> n over 3 constraints
    express = []
    A = Matrix.dense([[1, -1, -1]])
    for i,j in pairs: 
        for k in range(j+1,n+1):
            ij = get_index(i,j,n)
            jk = get_index(j,k,n)
            ik = get_index(i,k,n)
            express.append(Expr.dot(A, Y.pick([[ij,jk], [ij,ik], [ik,jk]])))
    M.constraint("3-cycle", Expr.vstack(express), Domain.equalsTo(-1))
    
    #M.constraint("Symmetry", Y[0,0], Domain.equalsTo(1.0))

    
    ## Betweeness constraints
    # for i,j in pairs: 
    #     for k in range(j+1,n+1):
    #         M.constraint(Expr.sub(Expr.sub(Y.index(get_index(i,j,n), get_index(j,k,n)), Y.index(get_index(i,j,n),get_index(i,k,n))), Y.index(get_index(i,k,n), get_index(j,k,n))), Domain.equalsTo(-1.0))

    # Triangel constraints
    # A = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    # end_index = int(comb(n, 2))
    # for i in range(end_index-2):
    #     for j in range(i+1, end_index-1):
    #         for k in range(j+1, end_index):
    #             M.constraint(Expr.mul(A, Y.pick([[i,j],[i,k],[j,k]])), Domain.greaterThan([-1.0, -1.0, -1.0, -1.0]))
                
    return M
 
def solveNode(node: Node, instance, vl, lb):
    M = multi_obj_model(instance, Model(), vl, lb)
    Z = M.getVariable("Z")
    Y = Z.slice([1,1] , [instance.dim+1, instance.dim+1])
    if node.constraints:
        for i,j,k in node.constraints:  
                M.constraint(Y.index(i,j), Domain.equalsTo(k))
    start_time = time.time()
    M.solve()
    #print("---Relax %s seconds ---" % (time.time() - start_time))
    
    node.solved = True
    
    if M.getProblemStatus() != ProblemStatus.PrimalAndDualFeasible:
        M.dispose()
        return Solution(False, -1, -1, MAX_INT, MAX_INT, MAX_INT, [], [])
    else:
        lb = M.primalObjValue()
        Y_test = Y.level()[0:instance.dim]
        # if node.constraints:
        #     permutation = solution2permutation(Y_test, instance.n, node.constraints[0][2])
        #     Y_int = permutation2Y([j for i,j in permutation], instance, 1)
        # else:
        Y_rank = np.array(Y.level())
        if np.all(np.equal(np.mod(Y_rank, 1), 0)):
            print("Integer") 
        permutation = solution2permutation(Y_test, instance.n, 1, instance.pairs)
        Y_int = permutation2Y([j for i,j in permutation], instance, 1)
        #Y_test = Y.level()
        #Y_test = Y_int.reshape([instance.dim,instance.dim])
        #print(np.linalg.matrix_rank(Y_test))
        #print(np.allclose(Y_test, Y_test.T, atol=1e-8))
        #test = instance.K-(sum(instance.cost_matrix1.flatten()*Y.level().flatten()))
        obj1 = instance.K-(sum(instance.cost_matrix1.flatten()*Y_int.flatten()))
        obj2 = instance.K2 - sum(instance.cost_matrix2.flatten()*Y_int.flatten())
        #test = sum(instance.cost_matrix2.flatten()*Y.level().flatten())
        M.dispose()
        return Solution(True, node.index, node.level, lb, obj1, obj2, permutation, Y_int) 

def solveNode2(node, instance, vl, lb):
    ## Initialize the model
    M = multi_obj_model(instance, Model(), vl, lb)
    Z = M.getVariable("Z")
    Y = Z.slice([1,1] , [instance.dim+1, instance.dim+1])
    
    ## Add the branching constraints to the model
    if node.constraints:  
        M.constraint(Y.pick([[i,j] for i,j,k in node.constraints]), Domain.equalsTo([k for i,j,k in node.constraints]))

    ## Solve the model
    #start_time = time.time()
    M.solve()
    # try:
    #     print("Without triangle:", M.primalObjValue())
    # except:
    #     pass
    
    ## Commands to get a single constraint of a stack of constraints and the primal value of the constraint
    #print(M.getConstraint("3-cycle").index([10]).remove())
     
    # print(constraint_help.index([0]).level())
    
    #print("---Relax %s seconds ---" % (time.time() - start_time))
    
    # Code for the separation of the triangle constraints TODO needs further improvement 
    # if M.getProblemStatus() == ProblemStatus.PrimalAndDualFeasible:
    #     ## Get the triangle constraints
    #     T = Matrix.dense([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    #     triangle = separate_triangle(Y.level(), instance, 0)
    #     #triangle.sort(key=lambda x: x[4])
    #     #variables = Var.hstack([Y.pick([[i,j], [i,k], [j,k]]) for i,j,k,row,result in triangle[0:min(100, len(triangle))]])
    #     variables = Var.hstack([Y.pick([[i,j], [i,k], [j,k]]) for i,j,k,row,result in triangle])
    #     M.constraint("Triangle", Expr.mul(T, variables), Domain.greaterThan(-1.0))
    #     #count = 0
    #     # for i,j,k,row,result in triangle[0:min(400, len(triangle))]:
    #     #     count += 1
    #     #     #print(type([Y.index(i,j), Y.index(i,k), Y.index(j,k)]))
    #     #     M.constraint(Expr.add(Expr.add(row[0]*Y.index(i,j), row[1]*Y.index(i,k)), row[2]*Y.index(j,k)), Domain.greaterThan(-1.0))
    #     # #print("Number of constraints added:", count)
    #     #start_time = time.time()
    #     M.solve()
    #     #print("---Triangle %s seconds ---" % (time.time() - start_time))
    #     try:
    #         const_values = np.array(M.getConstraint("Triangle").level())
    #         #print(f"{np.count_nonzero(const_values <= -1 + EPSOPT)} tight of {len(const_values)} constraints")
            
    #     except:
    #         pass
            
    ## Check if the relaxation is primal and dual feasible
    if M.getProblemStatus() != ProblemStatus.PrimalAndDualFeasible:
        M.dispose()
        return Solution(False, node.index, node.level, 0, MAX_INT, MAX_INT, [], [])
    else:
        # Return the solution
        Y_sol = Y.level().copy()
        relax_sol = M.primalObjValue()
        M.dispose()
        return Solution(True, node.index, node.level, relax_sol, MAX_INT, MAX_INT, [], Y_sol)

## Solve the root node with the separation of the triangle constraints
def solveRootSep(node, instance, vl, lb, constraints, previous_solution):
    ## Initialize the model
    M = multi_obj_model(instance, Model(), vl, lb)
    Z = M.getVariable("Z")
    Y = Z.slice([1,1] , [instance.dim+1, instance.dim+1])
    
    ## Solve the model
    #start_time = time.time()
    M.solve()
    #print(M.getSolverIntInfo("optNumcon")) # -> Always after the solve, gives the number of constraints in the model
    #print("---Relax %s seconds ---" % (time.time() - start_time))
    
    ## Speparation of the triangle constraints
    # triangle = separate_triangle(Y.level(), instance, 0)
    
    ## solve again
    
    ## While num_of_tight < num_of_constraints
    ## Check which constraints are tight and remove the others
    ## Separate for violated constraints
    ## Add the violated constraints to the model
    ## Solve the model again
            
    # Check if the relaxation is primal and dual feasible
    if M.getProblemStatus() != ProblemStatus.PrimalAndDualFeasible:
        M.dispose()
        return Solution(False, node.index, node.level, 0, MAX_INT, MAX_INT, [], [])
    else:
        # Return the variable matrix Y
        Y_sol = Y.level().copy()
        relax_sol = M.primalObjValue()
        M.dispose()
        return Solution(True, node.index, node.level, relax_sol, MAX_INT, MAX_INT, [], Y_sol)


def heuristic(Y, instance, vl):
    Y_shaped = Y.reshape(instance.dim, instance.dim)
    Y_int_opt = []
    Y_int_obj = MAX_INT
    
    for i in range(0, instance.dim):
        
        omega_values = []
        for k in range(1,instance.n+1):
            omega = (instance.n+1)/2
            for l in range(1, instance.n+1):
                if k == l:
                    continue
                if k > l:
                    omega -= Y[get_index(l,k,instance.n)]/2
                elif k < l:
                    omega += Y[get_index(k,l,instance.n)]/2
            omega_values.append(omega)
        permutation_helper = list(zip(omega_values, range(1, instance.n+1)))
        permutation_helper.sort(reverse=True)
        permutation = [j for i,j in permutation_helper]
        
        Y_help = []
        for i,j in instance.pairs:
            if permutation.index(i) < permutation.index(j):
                Y_help.append(-1)
            else:
                Y_help.append(1)
                
        Y_int = np.outer(np.array(Y_help),np.array(Y_help).T)
        
        if calculateObjective(Y_int, instance, 2) <= vl:
            if calculateObjective(Y_int, instance, 1) < Y_int_obj:
                Y_int_obj = calculateObjective(Y_int, instance, 1)
                Y_int_opt = Y_int
        else:
            if calculateObjective(Y_int, instance, 1) < Y_int_obj:
                Y_int_obj = calculateObjective(Y_int, instance, 1)
                Y_int_opt = Y_int
                
    return Y_int_opt
            
        
        
        
    ## OLD
    # # Create the permutation
    # R_values = np.zeros((instance.n,instance.n-1))    
    # for i in range(len(instance.pairs)):
    #     x=instance.pairs[i][1]-2
    #     y=instance.pairs[i][0]-1
    #     R_values[y,x] = Y[i]
    #     R_values[x+1,y] = -Y[i]
    
    # # Sort the permutation
    # p_help = R_values.sum(axis = 1)
    # p_values = list(zip(p_help, range(1, instance.n+1)))
    # p_values.sort(reverse=True)
    # permutation = [j for i,j in p_values]

    # Create the Y matrix with only 1 and -1 from the permutation
    # Y_help = []
    # for i,j in instance.pairs:
    #     if permutation.index(i) < permutation.index(j):
    #         Y_help.append(-1)
    #     else:
    #         Y_help.append(1)
            
    # Y_int = np.outer(np.array(Y_help),np.array(Y_help).T)
    
    
    #return Y_int

        
def checkIntegerFeasible(solution, vl):
    if solution.feasible and solution.obj2+EPSILON <= vl and solution.obj2 >= 0:
        return True
    else: 
        return False
    
def checkOptimal(incumbent, global_ub, global_lb):
    if incumbent.feasible and incumbent.obj1 - global_lb - EPSOPT< 0.5:
        return True
    else:
        return False
        
def branching(openNodes, remainingVar):
    # Sort the openNodes list to get the node with the lowest lower bound
    openNodes.sort()
    currentNode = openNodes[0]
    
    # TODO Is this still necessary?
    if currentNode.level < len(remainingVar):
        # Add the branching constraints
        i,j = remainingVar[currentNode.level]
        constraint1 = currentNode.constraints.copy()
        constraint1.append((i,j,-1))
        constraint2 = currentNode.constraints.copy()
        constraint2.append((i,j,1))
        level = currentNode.level
        
        # Build the nodes
        node1 = Node(currentNode.index+1, level+1, currentNode.lb, currentNode.ub, constraint1, False)
        node2 = Node(currentNode.index+2, level+1, currentNode.lb, currentNode.ub, constraint2, False)
        
        # Add the nodes to the openNodes list
        openNodes.append(node1)
        openNodes.append(node2)
    return openNodes
    
# TODO: Check if this is still necessary
def initRemainingVar(dim):
    remainingVar = []
    # for i in range(0, dim):
    #     for j in range(i+1, dim):
    #         remainingVar.append((i,j))
    for j in range(0, dim):
        remainingVar.append((0,j))        
    
            
    return remainingVar

def updateNode(node, solution, vl):
    if solution.feasible:
        node.lb = solution.lb
        if checkIntegerFeasible(solution, vl):
            node.ub = solution.obj1
        else:
            pass
            # node.ub = MAX_INT
                
def update_bounds(openNodes, global_lb, global_ub):
    # Sort the openNodes list to get the node with the lowest lower bound
    sortedNodes = sorted(openNodes)
    
    # Calculate the new lower bound
    betterLB = math.ceil(sortedNodes[0].lb * 2.0) / 2.0
    global_lb = betterLB
    
    # Calculate the new upper bound
    for node in openNodes:
        if node.ub < global_ub:
            global_ub = node.ub
            
    return global_lb, global_ub
            
def bNb(instance, vl, lb):
    incumbent = Solution(False, -1, -1, 0, MAX_INT, MAX_INT, [], [])
    rootNode = Node(0, 0, 0, MAX_INT, [], False)
    var2branch = initRemainingVar(instance.dim)
    #random.shuffle(var2branch)
    #print(var2branch)
    openNodes = [rootNode]
    #currentSol = solveNode(rootNode, instance, vl, lb)
    
    global_lb = lb
    global_ub = MAX_INT
    
    start_time = time.time()
    currentSol = solveNode(rootNode, instance, vl, global_lb)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    if currentSol.feasible:
        if checkIntegerFeasible(currentSol, vl): # True if primal/dual feasible and integer feasible
            incumbent = currentSol
            global_ub = currentSol.obj1
            rootNode.ub = currentSol.obj1
        global_lb = currentSol.lb
        rootNode.lb = currentSol.lb   
    
    if checkOptimal(incumbent, global_ub, global_lb):
        print("Root is optimal")
        return incumbent
    
    
    
    # while from here
    while openNodes:
    # Branching
        openNodes = branching(openNodes, var2branch)
        if len(openNodes) < 2:
            return incumbent
        node1 = openNodes[-1]
        node2 = openNodes[-2]
        solution1 = solveNode(node1, instance, vl, global_lb)
        print("Node1:", solution1.obj1, solution1.obj2, node1.index)
        if solution1.feasible:
            updateNode(node1, solution1, vl)
            if checkIntegerFeasible(solution1, vl):
                if solution1.obj1 < incumbent.obj1:
                    incumbent = solution1
                incumbent = solution1
        else:
            openNodes.remove(node1)
            
        solution2 = solveNode(node2, instance, vl, global_lb)
        print("Node2:", solution2.obj1, solution2.obj2, node2.index)
        if solution2.feasible:
            updateNode(node2, solution2, vl)
            if checkIntegerFeasible(solution2, vl):
                if solution2.obj1 < incumbent.obj1:
                    incumbent = solution2
        else:
            openNodes.remove(node2)
        
        #print(node1.lb, node2.lb, global_lb)
    
        # Remove the branched node from the openNodes list
        
        #print("Before pop:",[(x.lb, x.index, x.level) for x in openNodes])
        
        print([(x.lb, x.index) for x in openNodes])
        
        openNodes.pop(0)
        #print(solution1.lb, solution2.lb)
        
        # Update the global bounds
        if openNodes:
            global_lb, global_ub = update_bounds(openNodes, global_lb, global_ub)
            for node in openNodes:
                if node.lb > global_ub:
                    openNodes.remove(node)
        
        # Check if the incumbent is optimal
        if checkOptimal(incumbent, global_ub, global_lb):
            print("Incumbent is optimal")
            return incumbent      
        
        #print(global_lb, global_ub)
        #print("")
    # Repeat the process    
        print("Incumbent", incumbent.obj1, incumbent.obj2)
        print("##########################")
    print("No more nodes to explore. Returning incumbent.", global_lb, global_ub)
    return incumbent 
    
def calculateObjective(Y, instance, obj):
    if obj == 1:
        return instance.K - np.inner(instance.cost_matrix1.flatten(), Y.flatten())
    else:
        return instance.K2 - np.inner(instance.cost_matrix2.flatten(), Y.flatten())
    
def branch_and_bound2(instance, vl):
    ## Initialize the tree
    ## Initialize the incumbent solution
    bNb_time = time.time()
    incumbent = Solution(False, -1, -1, 0, MAX_INT, MAX_INT, [], [])
    solver_time = []
    
    ## Initialize the root node
    openNodes = [Node(index=0, level=0, lb=0, ub=MAX_INT, constraints=[], solved=False, Y=[])]
    rootNode = openNodes[0]
    
    ## Set the global tree bounds
    global_lb = 0
    global_ub = MAX_INT
    global_obj2 = MAX_INT
    
    number_of_nodes = 1
    
    ## Solve the root node of the tree
    start_time = time.time()
    root_sol = solveNode2(rootNode, instance, vl, global_lb)
    solver_time.append(time.time() - start_time)
    
    ## Solve root node
    if not root_sol.feasible:
        return incumbent
    else:
        ## Update the root node
        rootNode.Y = root_sol.Y
        rootNode.solved = True
        rootNode.lb = root_sol.lb
        global_lb = root_sol.lb   
        
        # TODO In general, try to only have a openNodes list, nodes and solution, maybe a processed nodes list
        
        ## Calculate the heuristic solution of the root node and the objective values
        #heur_Y = heuristic(root_sol.Y, instance)
        heur_Y = heuristic(root_sol.Y, instance,vl) 
        obj1 = calculateObjective(heur_Y, instance, 1)
        obj2 = calculateObjective(heur_Y, instance, 2)
        
        if obj2 <= vl - EPSILON:            
            # As root, set the incumbent to the heuristic solution and the bounds to the relaxed solution
            global_ub = obj1
            incumbent = Solution(True, 0, 0, global_lb, obj1, obj2, [], heur_Y)

            if global_ub - global_lb < EPSILON:
                return incumbent
        
    ## Start with the branching process
    while openNodes:
        ## Get node to branch on
        branchNode = get_branching_node(openNodes)

        ## Get variable to branch on
        var_branch = get_branching_variable(branchNode, instance)
        
        ## As long as there exists a variable to branch on, branch
        if var_branch != -1:
            ## Branch on the variable
            openNodes = branching2(openNodes, branchNode, var_branch)
            
            ## Pick the last two nodes in the openNodes list -> they are the branched nodes
            nodesToProcess = [openNodes[-1], openNodes[-2]]
            
            ## Process the nodes
            for node in nodesToProcess:
                number_of_nodes += 1
                solver_start = time.time()
                solution = solveNode2(node, instance, vl, global_lb)
                solver_time.append(time.time() - solver_start)
                
                node.solved = True
                node.Y = solution.Y
                node.lb = solution.lb
                if not solution.feasible:
                    openNodes.remove(node)
                else:
                    #heurSol = heuristic(solution.Y, instance)
                    heurSol = heuristic(solution.Y, instance, vl)
                    obj1 = calculateObjective(heurSol, instance, 1)
                    obj2 = calculateObjective(heurSol, instance, 2)
                    if obj2 <= vl - EPSILON:
                        node.ub = obj1
                        if obj1 < incumbent.obj1:
                            incumbent = Solution(True, node.index, node.level, node.lb, obj1, obj2, [], heurSol)
                            global_obj2 = obj2
            
            ## Remove the branched node from the openNodes list
            openNodes.remove(branchNode)
            
            ## Update the global bounds and remove nodes that are infeasible by bounds
            if openNodes:
                global_lb, global_ub = update_bounds(openNodes, global_lb, global_ub)
                for node in openNodes:
                    if node.lb > global_ub:
                        openNodes.remove(node)
            
            ## Check if the incumbent is optimal
            if global_ub - global_lb < EPSILON:
                break
            
            #print("Gap", global_ub - global_lb)
            
    print("Total solver time: %s seconds" % (sum(solver_time)))
    print("Total bNb time: %s seconds" % (time.time() - bNb_time - sum(solver_time)))
    print("Number of nodes processed:", number_of_nodes)
    
    return incumbent
          
def branching2(openNodes, branchNode, mostfrac):
    # Create two new nodes with the branching constraint
    constraint1 = branchNode.constraints.copy()
    constraint1.append((0, mostfrac, -1))
    constraint2 = branchNode.constraints.copy()
    constraint2.append((0, mostfrac, 1))
    node1 = Node(index=branchNode.index+1, level=branchNode.level+1, lb=branchNode.lb, ub=branchNode.ub, constraints=constraint1, Y=[], solved=False)
    node2 = Node(index=branchNode.index+2, level=branchNode.level+1, lb=branchNode.lb, ub=branchNode.ub, constraints=constraint2, Y=[], solved=False)
    openNodes.append(node1)
    openNodes.append(node2)
    return openNodes

def most_fractional(Y):
    variable = -1
    distance = 1
    for i in range(0, len(Y)):
        if abs(Y[i]) < distance:
            variable = i
            distance = abs(Y[i])
    return variable

def epsilon_constraint(instance):
    # Create a model
    vl = MAX_INT
    dominatedpoints = []
    feasible = True
    lb = 0
    time_list = []
    
    while feasible == True:
        #print("VL", vl)
        start_time = time.time()
        incumbent = branch_and_bound2(instance, vl)
        #incumbent = bNb(instance, vl, lb)
        feasible = incumbent.feasible
        obj1 = incumbent.obj1
        obj2 = incumbent.obj2
        vl = obj2
        #lb = obj1
        dominatedpoints.append((obj1, obj2))
        timer = time.time() - start_time
        time_list.append(timer)
        print("Total time: %s seconds" % (timer))
        print(f"Objective1: {obj1}, Objective2: {obj2} \n")
    return dominatedpoints, time_list
    
def permutation2Y(permutation, instance, switch):
    Y_help = []
    for i,j in instance.pairs:
        if permutation.index(i) < permutation.index(j):
            Y_help.append(-1*switch)
        else:
            Y_help.append(1*switch)
    Y = np.outer(np.array(Y_help),np.array(Y_help).T)
    return Y
  
def generate_cost_vector(n):
    print(n)
    costs = [np.random.randint(1, 10) for i in range(0,n)]
    with open(("cost_vector_"+ str(n) + ".txt"), 'w') as file:
        file.write(' '.join(map(str, costs)))
            
def read_cost_vector(file_path):
    with open(file_path, 'r') as file:
        costs = [*map(int, file.readline().split(" ")),]
    return costs
                        
def get_branching_node(openNodes):
    openNodes.sort()
    return openNodes[0]

def get_branching_variable(node, instance):
    #print(node.Y)
    return most_fractional(node.Y[0:instance.dim])
    
def separate_triangle(Y, instance, number):
    Y_reshape = Y.reshape([instance.dim,instance.dim])
    A = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    count = 0
    triangle_constraints = []
    # for i in range(0, instance.dim):
    #     for j in range(i+1, instance.dim):
    #         if Y_reshape[i,j] > 1-EPSOPT:
    #             rows = A[2:4]
    #         elif Y_reshape[i,j] < -1+EPSOPT:
    #             rows = A[0:2]
    #         else:
    #             rows = A[0:4]
    #         for k in range(j+1, instance.dim):
    #             for row in rows:
    #                 result = np.inner(row, [Y_reshape[i,j], Y_reshape[i,k], Y_reshape[j,k]])
    #                 if result + EPSOPT < -1:
    #                     count += 1
    #                     triangle_constraints.append((i,j,k,row,result))
    #                     if count == number:
    #                         return triangle_constraints  
    #print("In separation loop")  
    while count < number:               
        for pairOne in instance.pairs:
            for pairTwo in instance.pairs[1:]:
                for pairThree in instance.pairs[2:]:
                    i,j = pairOne
                    k,l = pairTwo
                    m,n = pairThree
                    if i < j and j < k:
                        continue
                    else:
                        for rows in A:
                            p1 = get_index(i,j,instance.n)
                            p2 = get_index(k,l,instance.n)
                            p3 = get_index(m,n,instance.n)
                            result = np.inner(rows, [Y_reshape[p1, p2], Y_reshape[p1, p3], Y_reshape[p2, p3]])
                            #print(result)
                            if result < -1:
                                count += 1
                                triangle_constraints.append((p1,p2,p3,rows,result)) 
    triangle_constraints.sort(key=lambda x: x[4])
    #print("Number of violated triangles:", count)
    #print("Out of separation loop")
    return triangle_constraints
        
def main(argv):
    ## Read the arguments
    inputfile, inputfile2 = argument_parser(argv)
      
    ## Set the path to the output file
    output_file = "plots/"+inputfile.split("/")[-1]+"_"+inputfile2.split("/")[-1]+".txt"
    
    ## Set the standard output to the output file and console
    sys.stdout = Logger(output_file)
    
    ## Print the names of the input files
    print('Input file 1 is', inputfile)
    print('Input file 2 is', inputfile2)
    print("")
    
    ## Read the instance
    instance = read_instance(inputfile, inputfile2)
    
    ## Run the epsilon constraint method
    start_time = time.time()
    domiantedpoints, time_list = epsilon_constraint(instance)
    domiantedpoints = domiantedpoints[0:-1]
    print(domiantedpoints)
    print("")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Mean time:", np.mean(time_list))

    plt.scatter(*zip(*domiantedpoints))
    plot_name = "plots/plot_"+inputfile.split("/")[-1]+"_"+inputfile2.split("/")[-1]+".pdf"
    plt.savefig(plot_name)

if __name__ == "__main__":
    main(sys.argv[1:])