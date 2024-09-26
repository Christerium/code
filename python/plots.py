import argparse
import pandas as pd

import matplotlib.pyplot as plt

def remove_weakly_dominated(data):
    # Initialize a list to store the indices of weakly dominated points
    weakly_dominated_indices = []

    # Iterate over each data point
    for i in range(len(data)):
        # Get the objective values of the current data point
        obj1_current = data.loc[i, 'OBJ1']
        obj2_current = data.loc[i, 'OBJ2']

        # Check if the current data point is weakly dominated by any other data point
        is_weakly_dominated = False
        for j in range(len(data)):
            # Skip the current data point
            if i == j:
                continue

            # Get the objective values of the other data point
            obj1_other = data.loc[j, 'OBJ1']
            obj2_other = data.loc[j, 'OBJ2']

            # Check if the current data point is weakly dominated
            if obj1_current >= obj1_other and obj2_current >= obj2_other:
                is_weakly_dominated = True
                break

        # If the current data point is weakly dominated, add its index to the list
        if is_weakly_dominated:
            weakly_dominated_indices.append(i)

    # Remove the weakly dominated points from the dataframe
    data = data.drop(weakly_dominated_indices)

    return data

# Create an argument parser
parser = argparse.ArgumentParser(description='Plot obj1 vs obj2 from a CSV file.')
parser.add_argument('-i', '--input', type=str, help='Input CSV file path')

# Parse the command-line arguments
args = parser.parse_args()

# Read the CSV file
path = "cluster_results/stats/"
filename = args.input+"_detailed.txt"
data = pd.read_csv(path+filename)

density = args.input.split("_")[2]
num_range = args.input.split("_")[3]



# Call the function to remove weakly dominated points
data = remove_weakly_dominated(data)

# Extract obj1 and obj2 columns
obj1 = data['OBJ1']
obj2 = data['OBJ2']

# Create a scatter plot
plt.scatter(obj1, obj2)

# Set axis labels
plt.xlabel('Objective 1', fontsize=14)
plt.ylabel('Objective 2', fontsize=14)
plt.title(f"D={float(density)/100}, K={num_range}", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([2000, 18000])
plt.ylim([2000, 18000])
plt.tight_layout()

# Show the plot
plt.savefig('cluster_results/stats/'+args.input+'.pdf')
#plt.show()