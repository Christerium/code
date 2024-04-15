import cvxpy as cp
import numpy as np
from scipy.special import comb

def read_instance(file_path):
    with open(file_path, 'r') as file:
        pairs = []
        costs = []
        lengths = []
        for line in file:
            if line[0] == "e":
                pairs.append((int(line.split()[1]), int(line.split()[2])))
                costs.append(int(line.split()[3]))
            elif line[0] == "l":
                lengths.append(int(line.split()[1]))
        return pairs, costs, lengths

def get_index(i, j, n):
    return int(n*(i-1)-(((i-1)*(i-1)+(i-1))/2)+(j-i)-1)

def define_problem(pairs, costs, lengths):
    
    n = len(lengths)
    
    # Variable definition
    vector_length = int(comb(n, 2))
    X = cp.Variable((vector_length, vector_length), symmetric=True, name="X")
    
    # Objective function
    K = np.sum(costs)*np.sum(lengths)/2
    total_cost = K
    for i,j in pairs:
        k_sum = 0
        #print(range(1, n+1)[5])
        for k in range(1, n+1):
            if k == i or k == j:
                continue
            elif k < i:   
                k_sum += lengths[k-1]*X[get_index(k,i,n)][get_index(k,j,n)]
            elif k < j:
                k_sum -= lengths[k-1]*X[get_index(i,k,n)][get_index(k,j,n)]
            else:
                #print(get_index(i,k,n), get_index(j,k,n))
                k_sum += lengths[k-1]*X[get_index(i,k,n)][get_index(j,k,n)]
        total_cost -= k_sum * costs[get_index(i,j,n)]/2
    
    objective = cp.Minimize(total_cost)


    constraints = []
    for i,j in pairs:
        for k in range(j+1, n+1):
            constraints.append(X[get_index(i,j,n)][get_index(j,k,n)] - X[get_index(i,j,n)][get_index(i,k,n)] - X[get_index(i,k,n)][get_index(j,k,n)] == -1)

    constraints.append(cp.diag(X) == np.ones(vector_length))
    constraints += [X >> 0]    

    problem = cp.Problem(objective, constraints)

    problem.solve()

    print("The optimal value is", problem.value)
    print("A solution X is")
    print(X.value)

def main():
    pairs, costs, lengths = read_instance("python/simmons_1.txt")
    
    define_problem(pairs, costs, lengths)

if __name__ == "__main__":
    main()