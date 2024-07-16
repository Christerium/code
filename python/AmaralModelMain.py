import time
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys, getopt
import numpy as np

#root_ub = 0

def argument_parser(argv):
    inputfile = ""
    inputfile2 = ""
    timelimit = 3600
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--ifile", help="Input file 1")
        parser.add_argument("-j", "--jfile", help="Input file 2")
        parser.add_argument("-t", "--timelimit", help="Time Limit")
        
        args = parser.parse_args()
        
        if args.ifile:
            inputfile = "instance/"+args.ifile
        if args.jfile:
            inputfile2 = "instance/"+args.jfile
        if args.timelimit:
            timelimit = args.timelimit
            
    except getopt.GetoptError as error:
        print("Error", error)
        sys.exit(2)
    
    return inputfile, inputfile2, timelimit


def read_instance(filename):
    with open(filename, 'r') as file:
        dimension = int(file.readline().strip())
        length = list(map(int, file.readline().strip().split(',')))
        
        cost_matrix = []
        for _ in range(dimension):
            row = list(map(int, file.readline().strip().split(',')))
            cost_matrix.append(row)
    
    costs = {}
    for i in range(dimension):
        for j in range(dimension):
            if i != j:
                costs[(i+1, j+1)] = cost_matrix[i][j]  
    
    return length, costs

def process_instance(filename1, filename2):
    l, c1 = read_instance(filename1)
    _, c2 = read_instance(filename2)
    n = len(l)
    return c1, c2, l , n
"""
def mycallback(m, where):
        global root_ub
        if where == GRB.Callback.MIP:
            print("NODES ", m.cbGet(GRB.Callback.MIP_NODCNT))
            if m.cbGet(GRB.Callback.MIP_NODCNT) <= 10:
                 #m._root_ub = m.cbGet(GRB.Callback.MIPNODE_OBJBST)
                root_ub = m.cbGet(GRB.Callback.MIP_OBJBST)
                print("HERE ", root_ub)
                 #m._root_lb = m.cbGet(GRB.Callback.MIPNODE_OBJBND)
                 
        if where == GRB.Callback.MIPSOL:
            print("NODES2 ", m.cbGet(GRB.Callback.MIPSOL_NODCNT))
            if m.cbGet(GRB.Callback.MIPSOL_NODCNT) <= 10:
                 #m._root_ub = m.cbGet(GRB.Callback.MIPNODE_OBJBST)
                root_ub = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
                print("HERE2 ", root_ub)
                 #m._root_lb = m.cbGet(GRB.Callback.MIPNODE_OBJBND)
                 
        if where == GRB.Callback.MIPNODE:
            print("NODES3 ", m.cbGet(GRB.Callback.MIPNODE_NODCNT))
            if m.cbGet(GRB.Callback.MIPNODE_NODCNT) <= 10:
                 #m._root_ub = m.cbGet(GRB.Callback.MIPNODE_OBJBST)
                root_ub = m.cbGet(GRB.Callback.MIPNODE_OBJBST)
                print("HERE3 ", root_ub)
                 #m._root_lb = m.cbGet(GRB.Callback.MIPNODE_OBJBND)
"""

# def getPermutation(sol):
#     for i in range(11):
#         for j in range(11):
#             for k in range(11):
#                 if sol[i][j][k] == 1:
#                     print(f"{i}, {j}, {k}")
#         #print("Solution: ", order)
        
def getPermutation(sol):
    n = len(sol)  # Assuming sol is a cubic matrix, n x n x n
    permutation = [-1] * n  # Initialize permutation with -1 or any placeholder to indicate unfilled positions
    
    facilities = set(range(n))
    
    fac_count_1 = []
    fac_count_2 = []
    fac_count_3 = []
    
    # Step 1: Find the start (i) and end (j) elements
    # This example assumes a simplistic approach and might need adjustments based on the specific problem constraints
    for facility in facilities:
        count_1 = 0
        count_2 = 0
        count_3 = 0
        for j in range(n):
            for k in range(n):
                if sol[j][k][facility] == 1:
                    # if facility == 8:
                    #     print(j, k)
                    count_1 += 1
                if sol[facility][j][k] == 1:
                    count_2 += 1
                if sol[j][facility][k] == 1:
                    count_3 += 1
        fac_count_1.append(count_1)
        fac_count_2.append(count_2)
        fac_count_3.append(count_3)
    
    print(fac_count_1)
    print(fac_count_2)
    print(fac_count_3)
    
    frontback = np.where(np.array(fac_count_3) == np.zeros(n))[0]
    if fac_count_2[frontback[0]] == 0:
        start = frontback[1]
        end = frontback[0]
        
    else:
        start = frontback[0]
        end = frontback[1]
    
    permutation[0] = start
    permutation[-1] = end
    facilities.remove(start)
    facilities.remove(end)
    #print(facilities)
 
    for fac in facilities:
        count = 0
        for j in range(n):
            if sol[fac][j][end] == 1:
                count += 1
        print(count)
        #print(n-1-count)
        permutation[n-2-count] = fac
    

        
    # Step 2: Fill the permutation array
    
    # for k in facilities:
    #     count = 0
    #     for j in range(n):
    #         if sol[start][j][k] == 1:
    #             print(start, j, k)
    #             count += 1
    #     permutation[count-1] = k
    #     print(count)
    #     #facilities.remove(k)
    
    
    
    # print("Start:", start, "End:", end)
    print("Permutation:", permutation)
    return permutation

# Example usage



def Amaral_Model(c1, c2, l, n, v, time_limit):         
    
    model_time = time.time()
    # Create a new GP instance
    m = gp.Model("SRFLP")

    # Add variables
    x = m.addVars(n, n, n, vtype=GRB.BINARY, name="x",  lb=0.0, ub=1.0)

    # Add constraint:
    m.addConstrs((x[i, j, k] + x[i, k, j] + x[j, i, k] == 1 for i in range(0,n) for j in range(i+1,n) for k in range(j+1,n) if (i < j and j < k)), name='7')

    m.addConstrs((-x[i, d, j] + x[i, d, k] + x[j, d, k] >= 0 for i in range(0,n) for j in range(i+1,n) for k in range(j+1,n) for d in range(n) if (i < j and j < k and d!=i and d!=j and d!=k)), name='8.1')
    m.addConstrs((x[i, d, j] - x[i, d, k] + x[j, d, k] >= 0 for i in range(n) for j in range(i+1,n) for k in range(j+1,n) for d in range(n) if (i < j and j < k and d!=i and d!=j and d!=k)), name='8.2')
    m.addConstrs((x[i, d, j] + x[i, d, k] - x[j, d, k] >= 0 for i in range(n) for j in range(i+1,n) for k in range(j+1,n) for d in range(n) if (i < j and j < k and d!=i and d!=j and d!=k)), name='8.3')

    m.addConstrs((x[i, d, j] + x[i, d, k] + x[j, d, k] <= 2 for i in range(n) for j in range(i+1,n) for k in range(j+1,n) for d in range(n) if (i < j and j < k and d!=i and d!=j and d!=k)), name='9')
    
    
    #print(type(float(c1[0, 0])*l[0]*x[0][1][2]))
    # Set objective_1 as main and esilon_constraint for objective_2 
    constant_1 = gp.quicksum(float(c1[i+1, j+1])*(l[i] + l[j]) for i in range(0,n-1) for j in range(i+1,n))/2
    objective_1 = gp.quicksum((float(c1[i+1, j+1])*l[k]*x[i, k, j]) for i in range(0,n-1) for j in range(i+1,n) for k in range(n) if (k != i and k != j))
    # objective_1 = gp.quicksum(((c1[i+1, j+1] * l[k]) - (c1[i+1, k+1] * l[j])) * x[i, j, k] for i in range(n) for j in range(i+1, n) for k in range(n) if k != i and k < j)
    # constant_1_1 = gp.quicksum((c1[i+1, j+1]/2) * (l[i] + l[j]) for i in range(n) for j in range(i+1, n))
    # constant_1_2 = gp.quicksum(c1[i+1, j+1] * l[k] for i in range(n) for j in range(i+1, n) for k in range(j+1, n))
    #print("\nConstant1: ", constant_1, constant_1_1+constant_1_2, "\n")
    
            
    # objective_2 = gp.quicksum(((c2[i+1, j+1] * l[k]) - (c2[i+1, k+1] * l[j])) * x[i, j, k] for i in range(n) for j in range(i+1, n) for k in range(n) if k != i and k < j)
    # constant_2_1 = gp.quicksum((c2[i+1, j+1]/2) * (l[i] + l[j]) for i in range(n) for j in range(i+1, n))
    # constant_2_2 = gp.quicksum(c2[i+1, j+1] * l[k] for i in range(n) for j in range(i+1, n) for k in range(j+1, n))
    
    constant_2 = gp.quicksum(c2[i+1, j+1]*(l[i] + l[j]) for i in range(0,n-1) for j in range(i+1,n))/2
    objective_2 = gp.quicksum(c2[i+1, j+1] * l[k] * x[i,k,j] for i in range(0,n-1) for j in range(i+1,n) for k in range(n) if (k != i and k != j))
    
    # obj1 = objective_1 + constant_1_1 + constant_1_2
    # obj2 = objective_2 + constant_2_1 + constant_2_2
    
    obj1 = objective_1 + constant_1
    obj2 = objective_2 + constant_2
    
    m.setObjective(obj1, GRB.MINIMIZE)
    
    
    # the epsilon-constriant
    epsilon = 1
    #m.addConstr(objective_2 + constant_2_1 + constant_2_2 <= v + constant_2_1 + constant_2_2 - epsilon)
    m.addConstr(obj2 <= v - epsilon)
    
    # Solve the problem
    # m.setParam(GRB.Param.TimeLimit, 5.0)
    #m.optimize(mycallback)

    
    time_remain = time_limit - (time.time() - model_time)
    
    if time_remain < 0:
        time_remain = 0

    try:
        m.setParam(GRB.Param.NodeLimit, 0)
        m.setParam('TimeLimit', time_remain)
        m.setParam(GRB.Param.Threads, 1)
        #m.setParam(GRB.Param.Presolve, 0)
        #m.setParam(GRB.Param.Cuts, 0)
        m.setParam(GRB.Param.Method, 2)
        m.optimize()
        if m.Status != GRB.INFEASIBLE:
            Rootub = m.ObjVal
            Rootlb = m.ObjBoundC
            print("Root Upper", Rootub)
            print("Root Lower", Rootlb)
            m.resetParams()
            time_remain = time_remain - (time.time() - model_time)
            if time_remain < 0:
                time_remain = 0
            m.setParam('TimeLimit', time_remain)
            #m.setParam(GRB.Param.Threads, 1)
            #m.setParam(GRB.Param.Presolve, 0)
            m.setParam(GRB.Param.Method, 2)
            m.optimize()
            BNB_Nodes = m.NodeCount
    except Exception as e:
        print("There is an error: ", e)
    
    
    #sol = [[[0 for k in range(11)] for j in range(11)] for i in range(11)]
# Assuming sol is filled according to the problem's conditions
    #print(getPermutation(x.X))
    
    #print("X", x.getAttr)
    
    # print("status:", GRB.OPTIMAL)
    # print("Root Upper", root_ub)
    # print("Node Count ", m.NodeCount)
    # if BNB_Nodes > 0:
    #      #Rootub = m._root_ub
    #     print("Root UB ", root_ub)
    #      #Rootlb = m._root_lb
    # else:
    #      #Rootub = m._root_ub
    #      #print("Root UB ", Rootub)
    #     print("Root UB ", root_ub)
    #     Rootub = 0
    #     Rootlb = 0
    
    
    
    # Get solution
    
    
    if m.Status == GRB.INFEASIBLE:
        print("No solution found")
        m.dispose()
        return None, None, None, GRB.INFEASIBLE, None, None, None, None
    elif m.Status == GRB.TIME_LIMIT:
        print("Time Limit reached")
        m.dispose()
        return None, None, None, GRB.INFEASIBLE, None, None, None, None
    else:
        obj1 = m.objVal
        obj2 = obj2.getValue()
        obj2_K = objective_2.getValue()
        status = m.Status
        m.dispose()
        return obj1, obj1, obj2_K, status, obj2, BNB_Nodes, Rootub, Rootlb
 


def main(argv):
    
    inputfile, inputfile2, timelimit = argument_parser(argv)
    
    
    output_file_basename = inputfile.split("/")[-1]+ "_" + inputfile2.split("/")[-1]
    
    print("Input file 1 is: ", inputfile)
    print("Input file 2 is: ", inputfile2)
    print("\n")
    
    
    
    # path = "D:/Research/Main code/generated/"
    # pairs = [("AC_8_30_5","AC_8_30_5_1"), ("AC_16_30_20","AC_16_30_20_1")]
    # file_pairs = [(path + part1, path + part2) for part1, part2 in pairs]  
    """
# Define the path to the directory
    path = "D:/Research/Main code/generated/"

# List all files in the directory
    files = os.listdir(path)

# Filter out the files that match the pattern and create pairs
    file_pairs = []
    for file in files:
        pair_file = file + "_1"
        if pair_file in files:
            file_pairs.append((os.path.join(path, file), os.path.join(path, pair_file)))
"""
    #instance_result = []
    
    #for i, (filename1, filename2)  in enumerate(file_pairs):
    #print(f"Running pair {i+1}: {filename1}, {filename2}")
    
    c1, c2, l, n = process_instance(inputfile, inputfile2)
    # the value of objective 2
    v = 99999999
    Start_Time = time.time()
    objective1 = []
    objective2 = []
    stats_list = []
    NDPoints = 0

    time_remain = int(timelimit)
    time_out = False

    
    obj_val, obj1, objective_2, status, obj2, BNB, rootub, rootlb = Amaral_Model(c1,c2,l,n,v, time_remain)
    
    print(obj_val, obj2)
    
    Total_Time_Point = time.time() - Start_Time
    time_remain -= (time.time() - Start_Time)
    
    while status != GRB.INFEASIBLE:
        Start_Time_Point = time.time() 
        
        #v = objective_2
        v = obj2
        NDPoints += 1
        print("flow1: ", obj1)
        print("flow2: ", obj2)
        print("Root Upper Bound:", rootub)
        print("Root lower Bound:", rootlb)
        print("ND-Point:", NDPoints)
        objective1.append(obj1)
        objective2.append(obj2)
        
        if rootub and rootlb and obj1 > 0:
            Rootgap = round(((rootub-rootlb)/rootub)*100, 2)
            RootgapOPT = round(((obj1-rootlb)/obj1)*100, 2)
        else:
            Rootgap = None
            RootgapOPT = None
            
        
        
        stats_list.append({'OBJ1': obj1, 'OBJ2': obj2, 'Rootgap_MIP%': Rootgap , 'Rootgap_OPT_MIP%': RootgapOPT, 'BNB_Nodes_MIP': BNB , 'Total_Time_MIP[s]' : Total_Time_Point})
        print("\n-----------------------------------------")
        try:
            obj_val, obj1, objective_2, status, obj2, BNB, rootub, rootlb = Amaral_Model(c1,c2,l,n,v, time_remain)
        except:
            status == GRB.INFEASIBLE
            
        End_time = time.time()
        Total_Time_Point = End_time - Start_Time_Point
        time_remain -= Total_Time_Point
        


    TotalTime = time.time() - Start_Time
    #instance_result.append({'Instance': inputfile.rsplit("/", 1)[-1], '#departments': n, '#ND-points': NDPoints, 't-MIP [s]':TotalTime})
    
    print("\n")
    print("Total running time:", TotalTime)
    
    
    
    
    #Table of final results 
    to_remove = []
    for i in range(0, len(stats_list)-1):
        if stats_list[i]["OBJ1"] == stats_list[i+1]["OBJ1"]:
            to_remove.append(stats_list[i])
            NDPoints-=1
    for i in to_remove:
        stats_list.remove(i)
        
    print("STAH, Instance, #departments, #ND-Points, t-MIP")
    print(f"STAT,{output_file_basename},{n},{NDPoints},{TotalTime}")
        
    dfResults = pd.DataFrame(stats_list)
    print(dfResults)
    dfResults.to_csv("./output/" + output_file_basename + ".csv", index=False)
    
    filename_caption = inputfile.rsplit("/", 1)[-1].replace("_"," ")
    #dfResults.to_latex("./output/" + output_file_basename +".tex", index=False, float_format="%.2f", caption="Non-dominated points statistics for instance "+filename_caption, label="tab:"+inputfile.rsplit('/', 1)[-1])
    #print(dfResults)
    
    #Plotting    
    #df = pd.DataFrame({'Objective1': objective1 ,'Objective2': objective2})
    #df.plot(kind='scatter', x= 'Objective1', y= 'Objective2' ) 
    plt.scatter(dfResults["OBJ1"], dfResults["OBJ2"])
    filename = "./output/" + output_file_basename+".pdf"
    print(filename)
    plt.savefig(filename) 
    plt.close()
    
    print("Plotting is done")
        
    # InstanceResults = pd.DataFrame(InstanceResults) 
    # InstanceResults.to_csv("InstanceResults.csv", index=False)
    # InstanceResults.to_latex("InstanceResults.tex", index=False, float_format="%.2f")
    # print(InstanceResults)
    
if __name__ == "__main__":
    main(sys.argv[1:])
