#import cvxpy as cp                  # For optimization
#import glob
import numpy as np                  # For matrix operations
from scipy.special import comb      # For binomnial coefficient
import mosek                        # Solver
import time                         # Time measurement
import sys, getopt                  # Command line arguments
from typing import NamedTuple       # C-like structure, but immutable (not changeable after creation)
import argparse                     # For command line arguments
from mosek.fusion import *          # For optimization
#import os
import matplotlib.pyplot as plt
import math
#import random

EPSILON = 0.5
MAX_INT = 999999999
EPSOPT = 1e-4
 
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
    print('Input file 1 is "', inputfile)
    print('Input file 2 is "', inputfile2)

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

    n = instance.n
    dim =  instance.dim

    Z = M.variable("Z", Domain.inPSDCone(dim+1))
    Y = Z.slice([1,1] , [dim+1, dim+1])

    C = Matrix.sparse(costs_matrix)
    C2 = Matrix.sparse(cost_matrix_2)

    # Objective function
    obj1 = Expr.sub(K, Expr.dot(C, Y))
    obj2 = Expr.sub(instance.K2, Expr.dot(C2, Y))
    M.objective(ObjectiveSense.Minimize, obj1)

    # Second objective constraint / Epsilon constraint
    M.constraint(obj2, Domain.lessThan(vl-EPSILON))
    
    # Diagonal equals to 1 constraints
    M.constraint("diag",Z.diag(), Domain.equalsTo(np.ones(dim+1))) # dim+1 constraints

    ## Betweeness constraints
    for i,j in pairs: 
        for k in range(j+1,n+1):
            # ij = get_index(i,j,n)
            # jk = get_index(j,k,n)
            # ik = get_index(i,k,n)
            # M.constraint(Expr.sub(Y.index(ij,jk), Expr.sub(Y.index(ij,ik), Y.index(ik,jk))), Domain.equalsTo(-1.0))
            M.constraint(Expr.sub(Expr.sub(Y.index(get_index(i,j,n), get_index(j,k,n)), Y.index(get_index(i,j,n),get_index(i,k,n))), Y.index(get_index(i,k,n), get_index(j,k,n))), Domain.equalsTo(-1.0))

    ## Triangel constraints
    # end_index = int(comb(n, 2))
    # for i in range(0, end_index):
    #     for j in range(0+1, end_index):
    #         for k in range(0+1, end_index):
    #             M.constraint(Expr.add(Expr.add(Y.index(i,j), Y.index(i,k)), Y.index(j,k)), Domain.greaterThan(-1.0))
    #             M.constraint(Expr.add(Expr.add(Y.index(i,j), Expr.neg(Y.index(i,k))), Expr.neg(Y.index(j,k))), Domain.greaterThan(-1.0))
    #             M.constraint(Expr.add(Expr.add(Expr.neg(Y.index(i,j)), Y.index(i,k)), Expr.neg(Y.index(j,k))), Domain.greaterThan(-1.0))
    #             M.constraint(Expr.add(Expr.add(Expr.neg(Y.index(i,j)), Expr.neg(Y.index(i,k))), Y.index(j,k)), Domain.greaterThan(-1.0))
            
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

# Rework of solveNode
def solveNode2(node, instance, vl, lb):
    # Initialize the model
    M = multi_obj_model(instance, Model(), vl, lb)
    Z = M.getVariable("Z")
    Y = Z.slice([1,1] , [instance.dim+1, instance.dim+1])
    
    # Add the branching constraints to the model
    if node.constraints:
        for i,j,k in node.constraints:  
                M.constraint(Y.index(i,j), Domain.equalsTo(k))
    
    # Solve the model
    M.solve()
    
    # Check if the relaxation is primal and dual feasible
    if M.getProblemStatus() != ProblemStatus.PrimalAndDualFeasible:
        M.dispose()
        #print("Relaxation is not primal and dual feasible, returning -1")
        return -1
    else:
        # Return the variable matrix Y
        #print("Relaxation obj1:", M.primalObjValue())
        #print("Relaxation obj2:", instance.K2 - sum(instance.cost_matrix2.flatten()*Y.level().flatten()))
        Y_sol = Y.level().copy()
        M.dispose()
        return Y_sol

def heuristic(Y, instance):
    # Create the permutation
    R_values = np.zeros((instance.n,instance.n-1))    
    for i in range(len(instance.pairs)):
        x=instance.pairs[i][1]-2
        y=instance.pairs[i][0]-1
        R_values[y,x] = Y[i]
        R_values[x+1,y] = -Y[i]
    
    p_help = R_values.sum(axis = 1)
    p_values = list(zip(p_help, range(1, instance.n+1)))
    p_values.sort(reverse=True)
    permutation = [j for i,j in p_values]
    
    # Create the Y matrix with only 1 and -1
    Y_help = []
    for i,j in instance.pairs:
        if permutation.index(i) < permutation.index(j):
            Y_help.append(-1)
        else:
            Y_help.append(1)
            
    Y_int = np.outer(np.array(Y_help),np.array(Y_help).T)
    
    # for i,j in instance.pairs:
    #     for k in range(j+1, instance.n+1):
    #         if Y_int[get_index(i,j,instance.n), get_index(j,k,instance.n)] - Y_int[get_index(i,j,instance.n), get_index(i,k,instance.n)] - Y_int[get_index(i,j,instance.n), get_index(j,k,instance.n)] != -1:
    #             print("Constraint violated")
    
    return Y_int
        
                

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
    # Sort the openNodes list to get the node with
    openNodes.sort()
    currentNode = openNodes[0]
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
    
    # Calculate the new lower bound
    betterLB = math.ceil(sortedNodes[0].lb * 2.0) / 2.0
    #print(betterLB, sortedNodes[0].lb)
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
    incumbent = Solution(False, -1, -1, 0, MAX_INT, MAX_INT, [], [])
    
    ## Initialize the root node
    openNodes = []
    rootNode = Node(0, 0, 0, MAX_INT, [], [], False)
    openNodes.append(rootNode)
    
    ## Set the global tree bounds
    global_lb = 0
    global_ub = MAX_INT
    global_obj2 = MAX_INT
    
    ## Solve the root node of the tree
    Y = solveNode2(rootNode, instance, vl, global_lb)
    
    try:
        if Y == -1:
            #print("Root node is infeasible")
            return (global_lb, global_ub, global_obj2)
    except:
        pass
    
    heur_sol = heuristic(Y, instance)    
    
    """ ## It needs to fulfill the constraints by construction
    # Check if the heuristic solution computed in the root node is feasible
    # Check if it violates the betweeness constraints
    for i,j in instance.pairs:
        for k in range(j+1,instance.n+1):
            i_j = get_index(i,j,instance.n)
            i_k = get_index(i,k,instance.n)
            j_k = get_index(j,k,instance.n)
            if heur_sol[i_j, j_k] - heur_sol[i_j, i_k] - heur_sol[i_k, j_k] == 1:
                print("Constraint violated")
    
    # Check if it violates the diagonal constraints
    if np.array_equal(np.diag(heur_sol), np.ones_like(heur_sol)):
        print("Diagonal constraint violated")
    
    # Check if it violates the rank one constraint
    if np.linalg.matrix_rank(heur_sol) != 1:
        print("Rank constraint violated")"""
    
    # Update the root node
    openNodes[0].Y = Y
    openNodes[0].solved = True
    openNodes[0].lb = calculateObjective(Y, instance, 1)
    global_lb = openNodes[0].lb
    
    # Check if the heuristic solution is feasible for the second objective function
    heur_obj2 = calculateObjective(heur_sol, instance, 2)
    if heur_obj2 <= vl - EPSILON:
        #print("Heuristic solution is feasible")
        # Calculate the objective value 1 of the heuristic solution
        heur_obj1 = calculateObjective(heur_sol, instance, 1)
        
        # As root, set the incumbent to the heuristic solution and the bounds to the relaxed solution
        global_ub = heur_obj1
        if global_ub - global_lb < EPSILON:
            #print("Root is optimal")
            print(global_lb, global_ub, heur_obj2)
            return (global_lb, global_ub, heur_obj2)
        incumbent = Solution(True, 0, 0, global_lb, heur_obj1, heur_obj2, [], heur_sol)
    
    ## Start with the branching process
    
    while openNodes:
    # Get variable to branch on
        openNodes.sort()
        node0 = openNodes[0]
        Y = node0.Y
        #print(Y)
        Y_first = Y[0:instance.dim]
        mostfrac = most_fractional(Y[0:instance.dim])
        if mostfrac == -1:
            #print("None fractional")
            print(global_lb, global_ub, global_obj2)
            #return (global_lb, global_ub, global_obj2)
        else:
            openNodes = branching2(openNodes, mostfrac)
            # Pick the last two nodes in the openNodes list
            node1 = openNodes[-1]
            node2 = openNodes[-2]
            
            # Solve the first node
            solution1 = solveNode2(node1, instance, vl, global_lb)
            node1.solved = True
            node1.Y = solution1
            #openNodes[-1].solved = True
            openNodes[-1].Y = solution1
            try:
                if solution1 == -1:
                    #print("Node1 is infeasible")
                    openNodes.remove(node1)
            except:
                # Calculate heuristic solution
                heur_sol1 = heuristic(solution1, instance)
                # Check if the solution is feasible for the second objective function
                lb = calculateObjective(solution1, instance, 1)
                node1.lb = lb
                #openNodes[-1].lb = lb
                obj2 = calculateObjective(heur_sol1, instance, 2)
                if obj2 <= vl - EPSILON:
                    # Obj2 is feasible
                    obj1 = calculateObjective(heur_sol1, instance, 1)
                    node1.ub = obj1
                    #openNodes[-1].ub = obj1
                    if obj1 < incumbent.obj1:
                        incumbent = Solution(True, node1.index, node1.level, lb, obj1, obj2, [], heur_sol1)
                        global_obj2 = obj2
                    

            
            # Solve the second node
            solution2 = solveNode2(node2, instance, vl, global_lb)
            node2.solved = True
            node2.Y = solution2
            #openNodes[-2].solved = True
            #openNodes[-2].Y = solution2
            try:
                if solution2 == -1:
                    #print("Node2 is infeasible")
                    openNodes.remove(node2)
            except:
                # Calculate heuristic solution
                heur_sol2 = heuristic(solution2, instance)
                # Check if the solution is feasible for the second objective function
                lb = calculateObjective(solution2, instance, 1)
                #openNodes[-2].lb = lb
                node2.lb = lb
                obj2 = calculateObjective(heur_sol2, instance, 2)
                if obj2 <= vl - EPSILON:
                    # Obj2 is feasible
                    lb = calculateObjective(solution2, instance, 1)
                    obj1 = calculateObjective(heur_sol2, instance, 1)
                    node2.ub = obj1
                    #openNodes[-2].ub = obj1
                    if obj1 < incumbent.obj1:
                        incumbent = Solution(True, node2.index, node2.level, lb, obj1, obj2, [], heur_sol2)
                        global_obj2 = obj2
            
        # Remove the branched node from the openNodes list
        openNodes.remove(node0)
        
        # Update the global bounds and remove nodes that are infeasible by bounds
        if openNodes:
            global_lb, global_ub = update_bounds(openNodes, global_lb, global_ub)
            for node in openNodes:
                if node.lb > global_ub:
                    openNodes.remove(node)
        else:
            pass
            #print("No more nodes to explore. Returning incumbent.", global_lb, global_ub)
            #return (global_lb, global_ub, global_obj2)
        
        # Check if the incumbent is optimal
        if global_ub - global_lb < EPSILON:
            pass
            #print("Incumbent is optimal")
            #return (global_lb, global_ub, global_obj2)
    
    #print("No more nodes to explore. Returning incumbent.", incumbent.obj1, incumbent.obj2)
    return incumbent
    
            
def branching2(openNodes, mostfrac):
    # Sort the openNodes list to get the node with the lowest lower bound
    openNodes.sort()
    # Pick the node with the lowest lower bound
    currentNode = openNodes[0]
    # Create two new nodes with the branching constraint
    constraint1 = currentNode.constraints.copy()
    constraint1.append((0, mostfrac, -1))
    constraint2 = currentNode.constraints.copy()
    constraint2.append((0, mostfrac, 1))
    node1 = Node(currentNode.index+1, currentNode.level+1, currentNode.lb, currentNode.ub, constraint1, [], False)
    node2 = Node(currentNode.index+2, currentNode.level+1, currentNode.lb, currentNode.ub, constraint2, [], False)
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
        print(obj1, obj2)
        timer = time.time() - start_time
        time_list.append(timer)
        print("--- %s seconds ---" % (timer))

        print("")
    return dominatedpoints, time_list


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
    
    inputfile, inputfile2 = argument_parser(argv)
    
    #inputfile = "code/instance/S/S8H"
    
    instance = read_instance(inputfile, inputfile2)
    
    
    start_time = time.time()
    
    # vl = 2012.5
    # solution = bNb(instance, vl, 0)
    # print(solution.obj1, solution.obj2)
    
    # vl = 1941.5
    # branch_and_bound2(instance, vl)

    domiantedpoints, time_list = epsilon_constraint(instance)
    domiantedpoints = domiantedpoints[0:-1]
    print(domiantedpoints)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Mean time:", np.mean(time_list))

    plt.scatter(*zip(*domiantedpoints))
    plot_name = "./plots/plot_"+inputfile.split("/")[-1]+"_"+inputfile2.split("/")[-1]+".pdf"
    plt.savefig(plot_name)
    #plt.show()
    
    #plt.plot(time_list)
    #plt.show()

    # print("##################")
    # M = Model()
    # vl = 2012.5
    # lb = 0
    # M = multi_obj_model(instance, Model(), vl, lb)
    
    
    # Z = M.getVariable("Z")
    # Y = Z.slice([1,1] , [instance.dim+1, instance.dim+1])
    
    # #M.constraint(Y.index(0,18), Domain.equalsTo(-1))
    # #M.constraint(Y.index(0,25), Domain.equalsTo(1))
    
    
    # #M.setLogHandler(sys.stdout)
    # M.solve()
    
    # print("SDP-Objective 1:", M.primalObjValue())
    
    # permutation = solution2permutation(Y.level()[0:instance.dim], instance.n, 1, instance.pairs)
    # #print(Y.level()[0:instance.dim])
    # Y_int = permutation2Y([j for i,j in permutation], instance, 1)
    # #print(Y_int[0])
    # print("Objective 1:", instance.K-(sum(instance.cost_matrix1.flatten()*Y_int.flatten())))
    # print("Objective 2:", instance.K2 - sum(instance.cost_matrix2.flatten()*Y_int.flatten()))
    # print("Obj2:", instance.K2 - sum(instance.cost_matrix2.flatten()*Y.level().flatten()))
    

if __name__ == "__main__":
    main(sys.argv[1:])