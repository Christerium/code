# Load necessary libraries
library(ggplot2)
library(dplyr)
library(argparse)

# Function to remove weakly dominated points
remove_weakly_dominated <- function(data) {
  # Convert OBJ1 and OBJ2 columns to numeric if they are not already
  view(data)
  data$OBJ1 <- as.numeric(as.character(data$OBJ1))
  data$OBJ2 <- as.numeric(as.character(data$OBJ2))
  
  
  # Check if OBJ1 and OBJ2 columns are numeric
  if (any(is.na(data$OBJ1)) || any(is.na(data$OBJ2))) {
    stop("OBJ1 and OBJ2 columns must be numeric and cannot contain NA values")
  }
  
  # Initialize a vector to store the indices of weakly dominated points
  weakly_dominated_indices <- c()
  
  # Iterate over each data point
  for (i in 1:nrow(data)) {
    # Get the objective values of the current data point
    obj1_current <- data[i, 'OBJ1']
    obj2_current <- data[i, 'OBJ2']
    
    # Print current values for debugging
    print(paste("Current point:", obj1_current, obj2_current))
    
    # Check if the current data point is weakly dominated by any other data point
    is_weakly_dominated <- FALSE
    for (j in 1:nrow(data)) {
      # Skip the current data point
      if (i == j) next
      
      # Get the objective values of the other data point
      obj1_other <- data[j, 'OBJ1']
      obj2_other <- data[j, 'OBJ2']
      
      # Print other values for debugging
      print(paste("Other point:", obj1_other, obj2_other))
      
      # Check if the current data point is weakly dominated
      if (obj1_current >= obj1_other && obj2_current >= obj2_other) {
        is_weakly_dominated <- TRUE
        break
      }
    }
    
    # If the current data point is weakly dominated, add its index to the vector
    if (is_weakly_dominated) {
      weakly_dominated_indices <- c(weakly_dominated_indices, i)
    }
  }
  
  # Remove the weakly dominated points from the dataframe
  if (length(weakly_dominated_indices) > 0) {
    data <- data[-weakly_dominated_indices, ]
  }
  
  return(data)
}

# Define file names and their corresponding labels
#files <- c("AC_12_30_20", "AC_12_50_20", "AC_12_70_20")
files <- c("AC_12_30_5", "AC_12_50_5", "AC_12_70_5")
labels <- c("0.3", "0.5", "0.7")

# Initialize an empty list to store data frames
data_list <- list()

# Read and process each file
for (i in seq_along(files)) {
  file <- files[i]
  label <- labels[i]
  filename <- paste0(file, "_", file, "_1_detailed.txt")
  path <- "cluster_results/stats"
  data <- read.csv(file.path(path, filename), header = TRUE)
  
  # Assign column names
  #colnames(data) <- c("OBJ1", "OBJ2", "col3", "col4", "col5", "col6", "col7")
  
  # Extract density and num_range from the file name
  input_parts <- strsplit(file, "_")[[1]]
  density <- as.numeric(input_parts[3])
  num_range <- input_parts[4]
  
  # Call the function to remove weakly dominated points
  data <- remove_weakly_dominated(data)
  
  # Add columns for density and num_range
  data$density <- density
  data$num_range <- num_range
  data$label <- label
  
  # Append the data frame to the list
  data_list[[file]] <- data
}

# Combine all data frames into one
combined_data <- bind_rows(data_list)

# Create a scatter plot
p <- ggplot(combined_data, aes(x = OBJ1, y = OBJ2, color = label)) +
  geom_point() +
  labs(
    x = 'f1',
    y = 'f2',
    title = 'n = 12; K = 5',
    color = 'D'
  ) +
  theme(
    text = element_text(size = 24),
    axis.title = element_text(size = 24),
    plot.title = element_text(size = 26)
  ) +
  scale_x_continuous(limits = c(350, 1400)) +
  scale_y_continuous(limits = c(350, 1400))
#   scale_x_continuous(limits = c(2000, 18000)) +
#   scale_y_continuous(limits = c(2000, 18000))

# Save the plot
ggsave(filename = 'cluster_results/stats/combined_plot_5.pdf', plot = p)













# Extract density and num_range from the input argument
input_parts <- strsplit(file, "_")[[1]]
density <- as.numeric(input_parts[3])
num_range <- input_parts[4]

# Call the function to remove weakly dominated points
data <- remove_weakly_dominated(data)

# Create a scatter plot
p <- ggplot(data, aes(x = OBJ1, y = OBJ2)) +
  geom_point() +
  labs(
    x = 'f1',
    y = 'f2',
    title = paste0("D=", density / 100, ", K=", num_range)
  ) +
  theme(
    text = element_text(size = 20),
    axis.title = element_text(size = 20),
    plot.title = element_text(size = 22)
  ) +
  scale_x_continuous(limits = c(350, 1400)) +
  scale_y_continuous(limits = c(350, 1400))
#    scale_x_continuous(limits = c(2000, 18000)) +
#    scale_y_continuous(limits = c(2000, 18000))

# Save the plot
ggsave(filename = paste0('cluster_results/stats/', "test", '.pdf'), plot = p)