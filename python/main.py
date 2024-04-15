import cvxpy as cp
import numpy as np

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

def main():
    pairs, costs, lengths = read_instance("python/simmons_1.txt")

    print(pairs)
    print(costs)
    print(lengths)


if __name__ == "__main__":
    main()