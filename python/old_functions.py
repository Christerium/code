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
        
    cost_matrix = define_costmatrix(pairs, costs, lengths)

    return pairs, costs, lengths, cost_matrix

# Define the optimization problem (Anjos and Vannelli 2008, p. 5) (not necessary anymor)
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

# Define the optimization problem (test with K-<C,Y>) (not necessary anymore)
def define_problem2(instance):    
    pairs, costs, lengths = instance.pairs, instance.costs, instance.lengths
    
    n = len(lengths)
    
    # Variable definition
    vector_length = int(comb(n, 2))
    X = cp.Variable((vector_length, vector_length), symmetric=True, name="X")
    
    # Objective function
    K = np.sum(costs)*np.sum(lengths)/2
    
    
    costs_matrix = define_costmatrix(pairs, costs, lengths)
    
    objective = cp.Minimize(K - cp.trace((costs_matrix @ X)))

    constraints = []
    for i,j in pairs:
        for k in range(j+1, n+1):
            constraints.append(X[get_index(i,j,n)][get_index(j,k,n)] - X[get_index(i,j,n)][get_index(i,k,n)] - X[get_index(i,k,n)][get_index(j,k,n)] == -1)
    

    """RHS = -(n-2)
    for i,j in pairs:
        constraint = 0
        for k in range(1, n):
            if k == i or k == j:
                continue
            else:
                constraint += X[get_index(i,j,n)][get_index(j,k,n)] - X[get_index(i,j,n)][get_index(i,k,n)] - X[get_index(i,k,n)][get_index(j,k,n)] 
        constraints.append(constraint >= RHS)   """
    
    constraints.append(cp.diag(X) == np.ones(vector_length))
    constraints += [X >> 0]    

    problem = cp.Problem(objective, constraints)

    print("Model read")

    problem.solve(solver = cp.MOSEK, verbose=True)

    print("The optimal value is", problem.value)
    #print("A solution X is")
    #print(X.value)

    return problem, X
