import cvxpy as cp                  # For optimization
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

EPSILON = 1
 
class Instance(NamedTuple):
    pairs: list
    costs: list
    lengths: list
    cost_matrix: np.matrix
    costs2: list

class Solution(NamedTuple):
    permutation: list
    objective: float
    Y: list

class Node(NamedTuple):
    lb: float
    ub: float
    constraints: list       # List of tupels (decisions)

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
    
    if os.path.exists("cost_vector_"+str(len(costs))+".txt"):
        costs2 = read_cost_vector("cost_vector_"+str(len(costs))+".txt")
    else:
        print("Cost vector does not exist")
        generate_cost_vector(len(costs))
        costs2 = read_cost_vector("cost_vector_"+str(len(costs))+".txt")

    return pairs, costs, lengths, cost_matrix, costs2

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

# Define the optimization problem with the triangle constraints and Z
def problem_sdp3(instance, M, sum_betweeness = False):
    pairs, costs, lengths = instance.pairs, instance.costs, instance.lengths    
    costs_matrix = define_costmatrix(pairs, costs, lengths)
    K = np.sum(costs)*np.sum(lengths)/2

    n = len(lengths)

    #M = Model("SDP3")

    #with Model("SDP3") as M:

    dim = len(pairs)
    dim2 = int(comb(n, 3))

    Z = M.variable("Z", Domain.inPSDCone(dim+1))
    Y = Z.slice([1,1] , [dim+1, dim+1])
    y = Z.slice([1,0], [1,dim+1])
    
    A2 = np.zeros((dim2, int(dim*dim)))
    l = 0
    for i,j in pairs:
        for k in range(j+1,n+1):
            ij, ik, jk = get_index(i,j,n), get_index(i,k,n), get_index(j,k,n)
            A2[l][(ij)*dim+jk] = 1
            A2[l][(ij)*dim+ik] = -1
            A2[l][(ik)*dim+jk] = -1
            l += 1
            
    A = Matrix.sparse(A2)

    C = Matrix.sparse(costs_matrix)
    e = Matrix.dense(np.ones((dim2,1))*-1)
    e2 = Matrix.dense(np.ones((3,1)))
    
    # Objective function
    M.objective(ObjectiveSense.Minimize, Expr.sub(K, Expr.dot(C, Y)))

    # Diagonal equals to 1 constraints
    M.constraint("diag",Expr.mulDiag(Y, Matrix.eye(dim)), Domain.equalsTo(1.0))

    # Betweenes constraints
    Y = Y.reshape([dim*dim, 1])
    M.constraint(Expr.mul(A, Y), Domain.equalsTo(e))

    # M constraints (triangle constraints)
    # T = [[-1, -1, -1],[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    # for i,j in pairs:
    #     for k in range(j+1,n+1):
    #         pass
    #         M.constraint(Expr.mul(T, Z.pick([[i,j],[i,k],[j,k]])), Domain.lessThan(1.0))
        
    return M

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
def solution2permutation(X, n):
    R_values = []
    for i in X:
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
def permutation2objective(p_values, instance):
    pairs, costs, costs2, lengths = instance.pairs, instance.costs, instance.costs2, instance.lengths
    objective = 0
    objective2 = 0
    for (x,y) in pairs:
        objective += costs[get_index(x,y,len(lengths))]*getLength(p_values, lengths, x, y)
        objective2 += costs2[get_index(x,y,len(lengths))]*getLength(p_values, lengths, x, y)
    return objective, objective2

def getRank(Y, dim):
    d = [0.0]*dim
    mosek.LinAlg.syeig(mosek.uplo.lo, dim, Y.level(), d)
    return sum([d[i] > 1e-6 for i in range(dim)])

def branch_and_bound(instance):
    optimal = False
    integer = False
    incumbent = Solution([], float("inf"))
    dim = len(instance.pairs)
    n = len(instance.lengths)

    incumbent_upper = float("inf")

    layers = []
    remaining_variables = []
    for i in range(0, dim):
        for j in range(i+1, dim):
            remaining_variables.append((i,j))

    open_nodes = []
    processed_nodes = []
    
    M = Model()
    M = problem_sdp3(instance, M, False)
    Z = M.getVariable("Z")
    Y = Z.slice([1,1] , [dim+1, dim+1])
    M.solve()
    # Check for feasibility
    if M.getProblemStatus() != ProblemStatus.PrimalAndDualFeasible:
        print("The problem is not feasible")
        optimal = False
    else:
        print("The problem is feasible")
        lb = M.primalObjValue()
        permutation = solution2permutation(Y.level()[0:dim], n)
        ub = permutation2objective(permutation, instance)
        if ub < incumbent_upper:
            incumbent_upper = ub
            incumbent = Solution(permutation, ub)
            print("New incumbent upper bound")
        if lb > incumbent_upper:
            print("Prune by bound")
        # Check for if lower bound is close enugh to upper bound
        if incumbent_upper - lb < 0.5:
            print("Branch cannot improve anymore")
        else:
            i,j = remaining_variables.pop(0)
            layers.append((i,j))
            open_nodes.append(Node(lb, ub, [-1]))
            open_nodes.append(Node(lb, ub, [1]))
            #print("\n Branching on", i, j)
            M.dispose()

    rootlb = lb

    while len(open_nodes) != 0:
        node = open_nodes.pop(0)
        M = Model()
        M = problem_sdp3(instance, M, False)
        Z = M.getVariable("Z")
        Y = Z.slice([1,1] , [dim+1, dim+1])
        #print(node.constraints)
        for i in range(len(node.constraints)):   #range(len(layers)):
            M.constraint(Y.index(layers[i][0], layers[i][1]), Domain.equalsTo(node.constraints[i]))
        M.solve()
        if M.getProblemStatus() != ProblemStatus.PrimalAndDualFeasible:
            #print("The problem is not feasible")
            optimal = False
        else:
            #print("The problem is feasible")
            lb = M.primalObjValue()
            #print(lb)
            permutation = solution2permutation(Y.level()[0:dim], n)
            ub = permutation2objective(permutation, instance)
            if ub < incumbent_upper:
                incumbent_upper = ub
                incumbent = Solution(permutation, ub)
                #print("New incumbent upper bound:", incumbent_upper)
            if lb > incumbent_upper:
                pass
                #print("Prune by bound")
            # Check for if lower bound is close enugh to upper bound
            if incumbent_upper - lb < 0.5:
                pass
                #print("\n Branch cannot improve anymore")
            else:
                i,j = remaining_variables.pop(0)
                layers.append((i,j))
                open_nodes.append(Node(lb, ub, node.constraints + [-1]))
                open_nodes.append(Node(lb, ub, node.constraints + [1]))
                #print("\n Branching on", i, j)

            processed_nodes.append(Node(lb, ub, node.constraints))
        #print(f"Remaining nodes: {len(open_nodes)}, Processed nodes: {len(processed_nodes)}, upper bound: {incumbent_upper}, gaptoroot: {(incumbent_upper-rootlb)/incumbent_upper}")
        M.dispose()

    print("Optimal solution is", incumbent_upper)
    print("Solution is", incumbent.permutation)

def multi_obj_model(instance2, M, vl):
    pairs, costs, costs2, lengths = instance2.pairs, instance2.costs, instance2.costs2, instance2.lengths    
    costs_matrix = define_costmatrix(pairs, costs, lengths)
    cost_matrix_2 = define_costmatrix(pairs, costs2, lengths)
    K = np.sum(costs)*np.sum(lengths)/2

    n = len(lengths)
    dim = len(pairs)
    dim2 = int(comb(n, 3))

    Z = M.variable("Z", Domain.inPSDCone(dim+1))
    Y = Z.slice([1,1] , [dim+1, dim+1])
    y = Z.slice([1,0], [1,dim+1])
    
    A2 = np.zeros((dim2, int(dim*dim)))
    l = 0
    for i,j in pairs:
        for k in range(j+1,n+1):
            ij, ik, jk = get_index(i,j,n), get_index(i,k,n), get_index(j,k,n)
            A2[l][(ij)*dim+jk] = 1
            A2[l][(ij)*dim+ik] = -1
            A2[l][(ik)*dim+jk] = -1
            l += 1
            
    A = Matrix.sparse(A2)

    C = Matrix.sparse(costs_matrix)
    e = Matrix.dense(np.ones((dim2,1))*-1)
    #e2 = Matrix.dense(np.ones((3,1)))
    
    # Objective function
    obj1 = Expr.sub(K, Expr.dot(C, Y))
    #obj1 = Expr.dot(C, Y)
    obj2 = Expr.dot(cost_matrix_2, Y)
    M.objective(ObjectiveSense.Minimize, obj1)

    # Second objective constraint
    M.constraint(obj2, Domain.lessThan(vl-EPSILON))
    
    # Diagonal equals to 1 constraints
    M.constraint("diag",Expr.mulDiag(Y, Matrix.eye(dim)), Domain.equalsTo(1.0))

    # Betweenes constraints
    Y = Y.reshape([dim*dim, 1])
    M.constraint(Expr.mul(A, Y), Domain.equalsTo(e))

    # M constraints (triangle constraints)
    # T = [[-1, -1, -1],[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    # for i,j in pairs:
    #     for k in range(j+1,n+1):
    #         pass
    #         M.constraint(Expr.mul(T, Z.pick([[i,j],[i,k],[j,k]])), Domain.lessThan(1.0))
        
    return M

def branch_and_bound2(instance, vl):
    optimal = False
    feasible = True
    obj1, obj2 = 99999999,99999999
    
    incumbent = Solution([], float("inf"), [])
    dim = len(instance.pairs)
    n = len(instance.lengths)

    incumbent_upper = float("inf")

    layers = []
    remaining_variables = []
    for i in range(0, dim):
        for j in range(i+1, dim):
            remaining_variables.append((i,j))

    open_nodes = []
    processed_nodes = []
    
    M = Model()
    M = multi_obj_model(instance, M, vl)
    Z = M.getVariable("Z")
    Y = Z.slice([1,1] , [dim+1, dim+1])
    M.solve()
    lb = 9999999999
    # Check for feasibility
    if M.getProblemStatus() != ProblemStatus.PrimalAndDualFeasible:
        #print("The problem is not feasible")
        optimal = False
        feasible = False
    else:
        #print("The problem is feasible")
        feasible = True
        lb = M.primalObjValue()
        permutation = solution2permutation(Y.level()[0:dim], n)
        ub, obj2 = permutation2objective(permutation, instance)
        if ub < incumbent_upper:
            incumbent_upper = ub
            incumbent = Solution(permutation, ub, Y.level())
            #print("New incumbent upper bound")
        if lb > incumbent_upper:
            print("Prune by bound")
        # Check for if lower bound is close enugh to upper bound
        if incumbent_upper - lb < 0.5:
            #print("Branch cannot improve anymore")
            pass
        else:
            i,j = remaining_variables.pop(0)
            layers.append((i,j))
            open_nodes.append(Node(lb, ub, [-1]))
            open_nodes.append(Node(lb, ub, [1]))
            #print("\n Branching on", i, j)
            M.dispose()

    #print(feasible)
    if feasible == True:
        rootlb = lb

        while len(open_nodes) != 0:
            node = open_nodes.pop(0)
            M = Model()
            M = multi_obj_model(instance, M, vl)
            Z = M.getVariable("Z")
            Y = Z.slice([1,1] , [dim+1, dim+1])
            #print(node.constraints)
            for i in range(len(node.constraints)):   #range(len(layers)):
                M.constraint(Y.index(layers[i][0], layers[i][1]), Domain.equalsTo(node.constraints[i]))
            M.solve()
            if M.getProblemStatus() != ProblemStatus.PrimalAndDualFeasible:
                #print("The problem is not feasible")
                optimal = False
            else:
                #print("The problem is feasible")
                lb = M.primalObjValue()
                #print(lb)
                permutation = solution2permutation(Y.level()[0:dim], n)
                ub, obj2 = permutation2objective(permutation, instance)
                if ub < incumbent_upper:
                    incumbent_upper = ub
                    incumbent = Solution(permutation, ub, Y.level())
                    #print("New incumbent upper bound:", incumbent_upper)
                if lb > incumbent_upper:
                    pass
                    #print("Prune by bound")
                # Check for if lower bound is close enugh to upper bound
                if incumbent_upper - lb < 0.5:
                    pass
                    #print("\n Branch cannot improve anymore")
                else:
                    i,j = remaining_variables.pop(0)
                    layers.append((i,j))
                    open_nodes.append(Node(lb, ub, node.constraints + [-1]))
                    open_nodes.append(Node(lb, ub, node.constraints + [1]))
                    #print("\n Branching on", i, j)

                processed_nodes.append(Node(lb, ub, node.constraints))
            #print(f"Remaining nodes: {len(open_nodes)}, Processed nodes: {len(processed_nodes)}, upper bound: {incumbent_upper}, gaptoroot: {(incumbent_upper-rootlb)/incumbent_upper}")
            M.dispose()

        #print("Optimal solution is", incumbent_upper)
        #print("Solution is", incumbent.permutation)
        
        obj1, obj2 = permutation2objective(incumbent.permutation, instance)
        
        cost_matrix2 = define_costmatrix(instance.pairs, instance.costs2, instance.lengths)
        cost_matrix = define_costmatrix(instance.pairs, instance.costs, instance.lengths)
        test = [round(elem, 0) for elem in incumbent.Y]

        K = np.sum(instance.costs)*np.sum(instance.lengths)/2
        obj1 = K - sum(cost_matrix.flatten()*test)
        obj2 = sum(cost_matrix2.flatten()*test)
        print("Obj1:", K - sum(cost_matrix.flatten()*test))
        print("Obj2:", sum(cost_matrix2.flatten()*test) )
        #print(incumbent.permutation)
        #print(test)
        
        
        return feasible, obj1, obj2
    else:
        return feasible, obj1, obj2

def epsilon_constraint(instance):
    # Create a model
    vl = 9999999999
    dominatedpoints = []
    feasible = True
    
    while feasible == True:
        feasible, obj1, obj2 = branch_and_bound2(instance, vl)
        vl = obj2
        dominatedpoints.append((obj1, obj2))
        print(obj1, obj2)
    return dominatedpoints


    # print solution
    
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
    
    instance = Instance(*read_instance(inputfile))
    
    #feasible, obj1, obj2 = branch_and_bound2(instance, 5000)
    
    #print(obj1, obj2)
    
    dominatedpoints = epsilon_constraint(instance)
    print(dominatedpoints)
    
    plt.scatter(*zip(*dominatedpoints))
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])