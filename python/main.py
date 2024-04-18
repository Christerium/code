import cvxpy as cp                  # For optimization
import numpy as np                  # For matrix operations
from scipy.special import comb      # For binomnial coefficient
import mosek                        # Solver
import time                         # Time measurement
import sys, getopt                  # Command line arguments
from typing import NamedTuple       # C-like structure, but immutable (not changeable after creation)


# ToDo:
# - Branch and bound algorithm
# - Think about branch and cut algorithm
# - Get the solution of the problem as a variable to work on. 
# - Implement the heuristic of the paper of Anjos, Kennings and Vannelli
# - Test it on different instances 
# - Programm a instance generate (I would do just a simple one, with random lenghts, costs and just wirting down all pairs.)
#   - Maybe do not even write all pairs, but just assume the order of the pairs and save the cost and length of the pairs.

class Instance(NamedTuple):
    pairs: list
    costs: list
    lengths: list

def argument_parser(argv):
    inputfile = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('main.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <inputfile> -o <outputfile> -g <instance size>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = "instance/"+arg
        elif opt in ("-g", "--generate"):
            generate_instance(int(arg))
            inputfile = "instance/instance.txt"

    print('Input file is "', inputfile)

    return inputfile

def generate_instance(n):
    with open("python/instance.txt", 'w') as file:
        for i in range(1, n+1):
            for j in range(i+1, n+1):
                file.write("e {} {} {} \n".format(i, j, np.random.randint(1, 10)))
        for i in range(1, n+1):
            file.write("l {} \n".format(np.random.randint(1, 10)))

# Read the instance file and return the pairs, costs and lengths
def read_instance(file_path):
    with open(file_path, 'r') as file:
        pairs = []
        costs = []
        lengths = []
        for line in file:
            if line[0] == "e":
                pairs.append((int(line.split()[1]), int(line.split()[2])))
                #costs.append(np.random.randint(1, 10))
                costs.append(int(line.split()[3]))
            elif line[0] == "l":
                lengths.append(int(line.split()[1]))
        return pairs, costs, lengths

# Get the index of the element in the matrix
def get_index(i, j, n):
    return int(n*(i-1)-(((i-1)*(i-1)+(i-1))/2)+(j-i)-1)

# Define the optimization problem (Anjos and Vannelli 2008, p. 5)
def define_problem(instance): 
    pairs, costs, lengths = instance.pairs, instance.costs, instance.lengths   
    n = len(lengths)

    # Variable definition
    vector_length = int(comb(n, 2))
    X = cp.Variable((vector_length, vector_length), symmetric=True, name="X")
    
    # Objective function
    K = np.sum(costs)*np.sum(lengths)/2
    total_cost = 0

    # Objective function
    for i,j in pairs:
        k_sum = 0
        cost = costs[get_index(i,j,n)]/2
        for k in range(1, n+1):
            if k == i or k == j:
                continue
            elif k < i:   
                k_sum += lengths[k-1]*X[get_index(k,i,n)][get_index(k,j,n)]
            elif k < j:
                k_sum -= lengths[k-1]*X[get_index(i,k,n)][get_index(k,j,n)]
            else:
                k_sum += lengths[k-1]*X[get_index(i,k,n)][get_index(j,k,n)]
        total_cost += k_sum * cost
    
    objective = cp.Minimize(K - total_cost)

    # Triangle constraints
    constraints = []
    for i,j in pairs:
        for k in range(j+1, n+1):
            constraints.append(X[get_index(i,j,n)][get_index(j,k,n)] - X[get_index(i,j,n)][get_index(i,k,n)] - X[get_index(i,k,n)][get_index(j,k,n)] == -1)

    # Diagonal constraint
    constraints.append(cp.diag(X) == np.ones(vector_length))
    
    # Positive semidefinite constraint
    constraints += [X >> 0]    

    problem = cp.Problem(objective, constraints)

    problem.solve(solver="MOSEK")

    print("The optimal value is", problem.value)

# Define the optimization problem (test with K-<C,Y>)
def define_problem2(instance):    
    pairs, costs, lengths = instance.pairs, instance.costs, instance.lengths
    
    n = len(lengths)
    
    # Variable definition
    vector_length = int(comb(n, 2))
    X = cp.Variable((vector_length, vector_length), symmetric=True, name="X")
    
    # Objective function
    K = np.sum(costs)*np.sum(lengths)/2
    
    start_time = time.time()
    costs_matrix = define_costmatrix(pairs, costs, lengths)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    objective = cp.Minimize(K - cp.trace((costs_matrix @ X)))

    constraints = []
    """ for i,j in pairs:
        for k in range(j+1, n+1):
            constraints.append(X[get_index(i,j,n)][get_index(j,k,n)] - X[get_index(i,j,n)][get_index(i,k,n)] - X[get_index(i,k,n)][get_index(j,k,n)] == -1)
    """

    RHS = -(n-2)
    for i,j in pairs:
        constraint = 0
        for k in range(1, n):
            if k == i or k == j:
                continue
            else:
                constraint += X[get_index(i,j,n)][get_index(j,k,n)] - X[get_index(i,j,n)][get_index(i,k,n)] - X[get_index(i,k,n)][get_index(j,k,n)] 
        constraints.append(constraint >= RHS)   
    
    constraints.append(cp.diag(X) == np.ones(vector_length))
    constraints += [X >> 0]    

    problem = cp.Problem(objective, constraints)

    problem.solve(solver="MOSEK")

    print("The optimal value is", problem.value)
    #print("A solution X is")
    #print(X.value)

    return problem, X

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


def main(argv):

    inputfile = argument_parser(argv)
    instance = Instance(*read_instance(inputfile))

    problem, X = define_problem2(instance)

    print(X[0][0].value)
    

if __name__ == "__main__":
    main(sys.argv[1:])