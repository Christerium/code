#import cvxpy as cp                  # For optimization
import glob
import numpy as np                  # For matrix operations
from scipy.special import comb      # For binomnial coefficient
import mosek                        # Solver
import time                         # Time measurement
import sys, getopt                  # Command line arguments
from typing import NamedTuple       # C-like structure, but immutable (not changeable after creation)
import argparse                     # For command line arguments
from mosek.fusion import *          # For optimization
import os
import matplotlib.pyplot as plt
import math
import random

EPSILON = 0.5
MAX_INT = 999999999
EPSOPT = 1e-4
 
class Node():
    def __init__(self, index, level, lb, ub, constraints, solved):
        self.index = index
        self.level = level
        self.lb = lb
        self.ub = ub
        self.constraints = constraints
        self.solved = solved
    
    def __str__(self):
        return f"Node {self.index}, level {self.level}, lb {self.lb}, ub {self.ub}, constraints {self.constraints}"

    def __lt__(self, other):
        return self.lb < other.lb

class Solution(NamedTuple):
    feasible: bool
    index: int
    level: int
    lb: float
    obj1: float
    obj2: float
    permutation: list
    Y: list
    
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
    K = np.sum(costs)*np.sum(lengths)/2
    
    if os.path.exists("cost_vector_"+str(len(costs))+".txt"):
        costs2 = read_cost_vector("cost_vector_"+str(len(costs))+".txt")
        cost_matrix2 = define_costmatrix(pairs, costs2, lengths)
    else:
        print("Cost vector does not exist")
        generate_cost_vector(len(costs))
        costs2 = read_cost_vector("cost_vector_"+str(len(costs))+".txt")
        cost_matrix2 = define_costmatrix(pairs, costs2, lengths)

    return Instance(pairs, lengths, costs, costs2, cost_matrix, cost_matrix2, len(pairs), len(lengths), K)

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
    pairs, costs, costs2, lengths = instance.pairs, instance.costs1, instance.costs2, instance.lengths    
    costs_matrix = instance.cost_matrix1
    cost_matrix_2 = instance.cost_matrix2
    K = instance.K

    n = instance.n
    dim =  instance.dim
    dim2 = int(comb(n, 3))

    Z = M.variable("Z", Domain.inPSDCone(dim+1))
    Y = Z.slice([1,1] , [dim+1, dim+1])
    y = Z.slice([1,0], [1,dim+1])
    
    # A2 = np.zeros((dim2, int(dim*dim)))
    # l = 0
    # for i,j in pairs:
    #     for k in range(j+1,n+1):
    #         ij, ik, jk = get_index(i,j,n), get_index(i,k,n), get_index(j,k,n)
    #         A2[l][(ij)*dim+jk] = 1
    #         A2[l][(ij)*dim+ik] = -1
    #         A2[l][(ik)*dim+jk] = -1
    #         l += 1
            
    # A = Matrix.sparse(A2)

    C = Matrix.sparse(costs_matrix)
    #e = Matrix.dense(np.ones((dim2,1))*-1)
    #e2 = Matrix.dense(np.ones((3,1)))
    
    # Objective function
    obj1 = Expr.sub(K, Expr.dot(C, Y))
    #obj1 = Expr.dot(C, Y)
    obj2 = Expr.dot(cost_matrix_2, Y)
    M.objective(ObjectiveSense.Minimize, obj1)

    # Second objective constraint
    #print("vl", vl)
    #print("Epsilon", EPSILON)
    M.constraint(obj2, Domain.lessThan(vl-EPSILON))
    
    ## Diagonal equals to 1 constraints
    M.constraint("diag",Expr.mulDiag(Z, Matrix.eye(dim+1)), Domain.equalsTo(1.0)) # dim+1 constraints

    # Diagonal equals to 1 constraints
    M.constraint("diag",Expr.mulDiag(Y, Matrix.eye(dim)), Domain.equalsTo(1.0))

    # Betweenes constraints
    Y = Y.reshape([dim*dim, 1])
    M.constraint(Expr.mul(A, Y), Domain.equalsTo(e))

    # M constraints (triangle constraints)
    # T = [[-1, -1, -1],[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    # for i in range(0, dim-3):
    #     for j in range(i+1, dim-2):
    #         for k in range(j+1, dim-1):
    #             #print(i,j,k)
    #             M.constraint(Expr.mul(T, Y.pick([[i,j],[i,k],[j,k]])), Domain.lessThan(1.0))
    
    ## Betweenes constraints (not feasible for large instances -> memory problems)
    # Y = Y.reshape([dim*dim, 1])
    # M.constraint(Expr.mul(A, Y), Domain.equalsTo(e))    # n over 3 constraints

    for i,j in pairs:
        for k in range(j+1,n+1):
            M.constraint(Expr.sub(Expr.sub(Y.index(get_index(i,j,n), get_index(i,k,n)), Y.index(get_index(i,j,n),get_index(i,k,n))), Y.index(get_index(i,k,n), get_index(j,k,n))), Domain.equalsTo(-1.0))

    # M constraints (triangle constraints)
    T = [[-1, -1, -1],[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    for i,j in pairs:
        for k in range(j+1,n+1):
            M.constraint(Expr.mul(T, Z.pick([[i,j],[i,k],[j,k]])), Domain.lessThan(1.0))

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
        obj2 = sum(instance.cost_matrix2.flatten()*Y_int.flatten())
        #test = sum(instance.cost_matrix2.flatten()*Y.level().flatten())
        M.dispose()
        return Solution(True, node.index, node.level, lb, obj1, obj2, permutation, Y_int) 

def checkIntegerFeasible(solution, vl):
    if solution.feasible and solution.obj2 <= vl-EPSILON and solution.obj2 >= 0:
        return True
    else: 
        return False
    
def checkOptimal(incumbent, global_ub, global_lb):
    if incumbent.feasible and incumbent.obj1 - global_lb - EPSOPT< 0.5:
        return True
    else:
        return False
        
def branching(openNodes, remainingVar):
    openNodes.sort()
    currentNode = openNodes[0]
    #print("Branching:", [(x.lb, x.index, x.level) for x in openNodes])
    #print(currentNode)
    #print(currentNode.level, len(remainingVar))
    if currentNode.level < len(remainingVar):
        
        #i,j = remainingVar.pop(0)
        #print(currentNode.level)
        i,j = remainingVar[currentNode.level]
        constraint1 = currentNode.constraints.copy()
        constraint1.append((i,j,-1))
        constraint2 = currentNode.constraints.copy()
        constraint2.append((i,j,1))
        level = currentNode.level
        node1 = Node(currentNode.index+1, level+1, currentNode.lb, currentNode.ub, constraint1, False)
        node2 = Node(currentNode.index+2, level+1, currentNode.lb, currentNode.ub, constraint2, False)
        openNodes.append(node1)
        openNodes.append(node2)
    return openNodes
    
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
    sortedNodes = sorted(openNodes)
    
    #print("Update:", [(x.lb, x.index, x.level) for x in sortedNodes])
    betterLB = math.ceil(sortedNodes[0].lb * 2.0) / 2.0
    #print(betterLB, sortedNodes[0].lb)
    global_lb = betterLB
    
    for node in openNodes:
        if node.ub < global_ub:
            global_ub = node.ub
    return global_lb, global_ub
            
def bNb(instance, vl, lb):
    incumbent = Solution(False, -1, -1, 0, MAX_INT, MAX_INT, [], [])
    rootNode = Node(0, 0, 0, MAX_INT, [], False)
    var2branch = initRemainingVar(instance.dim)
    random.shuffle(var2branch)
    #print(len(var2branch))
    openNodes = [rootNode]
    currentSol = solveNode(rootNode, instance, vl)
    
    global_lb = lb
    global_ub = MAX_INT
    
    start_time = time.time()
    currentSol = solveNode(rootNode, instance, vl, global_lb)
    #print("--- %s seconds ---" % (time.time() - start_time))
    
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
        if solution1.feasible:
            updateNode(node1, solution1, vl)
            if checkIntegerFeasible(solution1, vl):
                if solution1.obj1 < incumbent.obj1:
                    incumbent = solution1
                incumbent = solution1
        else:
            openNodes.remove(node1)
            
        solution2 = solveNode(node2, instance, vl, global_lb)
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
            return incumbent      
        
        #print(global_lb, global_ub)
        #print("")
    # Repeat the process    
    
    print("No more nodes to explore. Returning incumbent.", global_lb, global_ub)
    return incumbent 
    
def epsilon_constraint(instance):
    # Create a model
    vl = MAX_INT
    dominatedpoints = []
    feasible = True
    lb = 0
    
    while feasible == True:
        #print("VL", vl)
        #feasible, obj1, obj2 = branch_and_bound2(instance, vl)
        incumbent = bNb(instance, vl, lb)
        feasible = incumbent.feasible
        obj1 = incumbent.obj1
        obj2 = incumbent.obj2
        vl = obj2
        lb = obj1
        dominatedpoints.append((obj1, obj2))
        print(obj1, obj2)

        print("")
    return dominatedpoints


    # print solution
    
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
            
def main(argv):
    
    inputfile = argument_parser(argv)
    
    inputfile = "code/instance/S/S8H"
    
    instance = read_instance(inputfile)
    
    start_time = time.time()
    
    # vl = 9999999
    # solution = bNb(instance, vl)
    # print(solution.obj1, solution.obj2)
    
    domiantedpoints = epsilon_constraint(instance)
    domiantedpoints = domiantedpoints[0:-1]
    print(domiantedpoints)
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.scatter(*zip(*domiantedpoints))
    plt.show()
    
    # M = Model()
    # vl = 2401.5
    # lb = 0
    # M = multi_obj_model(instance, Model(), vl, lb)
    # Z = M.getVariable("Z")
    # Y = Z.slice([1,1] , [instance.dim+1, instance.dim+1])
    
    
    # M.setLogHandler(sys.stdout)
    # M.solve()
    
    # print(M.primalObjValue())
    
    # permutation = solution2permutation(Y.level()[0:instance.dim], instance.n, 1, instance.pairs)
    # Y_int = permutation2Y([j for i,j in permutation], instance, 1)
    # print(instance.K-(sum(instance.cost_matrix1.flatten()*Y_int.flatten())))
    # print(sum(instance.cost_matrix2.flatten()*Y_int.flatten()))
    

if __name__ == "__main__":
    main(sys.argv[1:])