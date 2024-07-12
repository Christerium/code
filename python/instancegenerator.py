import random
import numpy as np
import argparse
import os
import sys, getopt
import scipy as sp
import pathlib

def argument_parser(argv):
    num_departments = 10
    density = 0.5
    max_range = 5
    
    try:
        parser = argparse.ArgumentParser(description='Instance generator for the departmental clustering problem')
        parser.add_argument('-n', "--number", type=int, help='Number of departments')
        parser.add_argument('-d', "--density", type=int, help='Density of the instance')
        parser.add_argument('-r', "--range", type=int, help='Maximum range of the instance')

        args = parser.parse_args()
        
        if args.number:
            num_departments = args.number
        if args.density:
            density = args.density/100
        if args.range:
            max_range = args.range
    except getopt.GetoptError:
        print("instance_generator.py -n <number> -d <density> -r <range>")
        sys.exit(2)
        
    return num_departments, density, max_range
        
def generate_instance(num_departments, density, max_range):
    ## Set the seed for testing purposes
    #np.random.seed(0)
    
    ## Get the length of the array
    length = int(sp.special.binom(num_departments, 2))
    
    ## Generate the values
    values_cost = range(0, max_range+1)
    
    ## Generate the probability distribution
    prob_cost = [(1-density)] + [(density)/max_range] * (max_range)
    
    ## Generate the cost array
    random_cost = np.random.choice(values_cost, length, p=prob_cost)
    
    ## Transform the cost array into a cost matrix
    cost_matrix = np.zeros((num_departments, num_departments), dtype=int)
    for i in range(1, num_departments+1):
        for j in range(i+1, num_departments+1):
            n = num_departments
            index = int(n*(i-1)-(((i-1)*(i-1)+(i-1))/2)+(j-i)-1)
            cost_matrix[i-1][j-1] = random_cost[index]
            cost_matrix[j-1][i-1] = random_cost[index]
                
    ## Generate the length of the departments
    values_length = range(1, max_range+1)
    prob_length = [1/max_range] * max_range
    random_length = np.random.choice(values_length, num_departments, p=prob_length)
        
    return random_length, cost_matrix

def save_instance(lengths, cost_matrix, filename):
    with open('instance/' + filename, 'w') as file:
        file.write(str(len(lengths)) + '\n')
        file.write(','.join(lengths.astype(str)) + '\n')
        for i in range(len(cost_matrix)):
            file.write(','.join(cost_matrix[i].astype(str)) + '\n')
        
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    
    while os.path.exists('instance/'+ path):
        path = filename + '_' + str(counter) + extension
        counter += 1
    return path    

def main(argv):
    ## Parse the arguments
    num_departments, density, max_range = argument_parser(argv)   
    
    ## Generate the instance
    lengths, cost_matrix = generate_instance(num_departments, density, max_range)
    
    ## Save the instance to a file
    filename = 'AC_' + str(num_departments) + '_' + str(int(density*100)) + '_' + str(max_range)
    filename = uniquify(filename)
    save_instance(lengths, cost_matrix, filename)
    
if __name__ == "__main__":
    main(sys.argv[1:])