# Data configs
attributes = [i for i in range(29)]
continuous_variables = [5, 7, 11, 16, 17, 19, 21, 24, 28]
hexadecimal_variables = [8, 12, 14, 15, 23]
categorical_variables = [0, 1, 2, 3, 4, 6, 9, 10, 13, 18, 20, 22, 25, 26, 27]
string_variables = [1, 2, 4, 9, 10, 13, 18, 22, 25, 26]

remove_variables = [28, 17, 5]

# Experiment setup
number_of_variables = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
cross_validation = 10

# Model hyperparameters
learning_rate = [0.01, 0.05, 0.1]
depths = [3, 5, 7, 10, 15]
l2_leaf_reg = [1, 3, 5, 7, 9]
