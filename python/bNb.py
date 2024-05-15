from typing import NamedTuple       # C-like structure, but immutable (not changeable after creation)

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
    
def solveNode(node: Node, instance, vl):
    M = multi_obj_model(instance, Model(), vl)
    Z = M.getVariable("Z")
    Y = Z.slice([1,1] , [instance.dim+1, instance.dim+1])
    for i,j,k in node.constraints:  
            M.constraint(Y.index(i,j), Domain.equalsTo(k))
    M.solve()
    
    node.solved = True
    
    if M.getProblemStatus() != ProblemStatus.PrimalAndDualFeasible:
        return Solution(feasible, 0, MAX_INT, MAX_INT, [], [])
    else:
        lb = M.primalObjValue()
        permutation = solution2permutation(Y.level()[0:dim], instance.n)
        Y_int = permutation2solution(permutation, instance.n)
        obj1 = K-(sum(instance.cost_matrix1.flatten()*Y_int.flatten()))
        obj2 = sum(instance.cost_matrix2.flatten()*Y_int.flatten())
        return Solution(node.index, node.level, lb, obj1, obj2, permutation, Y_int)
        
def checkIntegerFeasible(solution, vl):
    if solution.feasible and solution.obj2 < vl-EPSILON:
        return True
    else: 
        return False
    
def updateLB(node, lb):
    pass

def updateUB():
    pass

def checkOptimal(incumbent, global_ub, global_lb):
    if incumbent.feasible:
        if incumbent.obj1 - global_lb < 0.5:
            return True
    else:
        return False
        
def branching(openNodes, remainingVar):
    openNodes.sort()
    currentNode = openNodes[0]
    i,j = remainingVar.pop(0)
    node1 = Node(currentNode.index+1, currentNode.level+1, currentNode.lb, currentNode.ub, currentNode.constraints.append((i,j,-1)), False)
    node2 = Node(currentNode.index+2, currentNode.level+1, currentNode.lb, currentNode.ub, currentNode.constraints.append((i,j,1)), False)
    openNodes.append(node1)
    openNodes.append(node2)
    return openNodes
    
def initRemainingVar(dim):
    remainingVar = []
    for i in range(0, dim):
        for j in range(i+1, dim):
            remainingVar.append((i,j))
    return remainingVar

def updateNode(node, solution):
    node.lb = solution.lb
    if checkIntegerFeasible(solution):
        node.ub = solution.obj1
    else:
        node.ub = MAX_INT
        
def update_bounds(openNodes, global_lb, global_ub):
    sortedNodes = sorted(openNodes)
    global_lb = sortedNodes[0].lb
    for node in openNodes:
        if node.ub < global_ub:
            global_ub = node.ub
    return global_lb, global_ub
            

def pruneNodes(openNodes):
    pass

def bNb(instance, vl):
    incumbent = Solution(-1, -1, 0, MAX_INT, MAX_INT, [], [])
    rootNode = Node(0, 0, 0, MAX_INT, [], False)
    var2branch = initRemainingVar(instance.dim)
    openNodes = [rootNode]
    currentSol = solveNode(rootNode, instance, vl)
    
    global_lb = 0
    
    if checkIntegerFeasible(currentSol, vl): # True if primal/dual feasible and integer feasible
        incumbent = currentSol
        global_lb = currentSol.lb
        global_ub = currentSol.obj1
        rootNode.ub = currentSol.obj1
        rootNode.lb = currentSol.lb   
    
    if checkOptimal(incumbent, global_ub, global_lb):
        return incumbent
    
    
    # while from here
    while openNodes:
    # Branching
        openNodes = branching(openNodes, var2branch)
        
        # Solve the two new nodes
        node1 = openNodes[-1]
        node2 = openNodes[-2]
        solution1 = solveNode(node1, instance, vl)
        updateNode(node1, solution1)
        if checkIntegerFeasible(solution1, vl):
            if solution1.obj1 < incumbent.obj1:
                incumbent = solution1
            incumbent = solution1
        solution2 = solveNode(node2, instance, vl)
        updateNode(node2, solution2)
        if checkIntegerFeasible(solution2, vl):
            if solution2.obj1 < incumbent.obj1:
                incumbent = solution2
                
    
        # Remove the branched node from the openNodes list
        openNodes.remove(0)
        
        # Update the global bounds
        global_lb, global_ub = update_bounds(openNodes)
        
        # Prune the open nodes & update the incumbent
        
        for node in openNodes:
            if node.lb > global_ub:
                openNodes.remove(node)
        
        # Check if the incumbent is optimal
        if checkOptimal(incumbent, global_ub, global_lb):
            return incumbent
        
    # Repeat the process    
    
    print("No more nodes to explore. Returning incumbent.")
    return incumbent 
    
   

        
    
    

    
    