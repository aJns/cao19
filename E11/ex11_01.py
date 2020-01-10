import matplotlib.pyplot as plt
import numpy as np
import time  # to track time

# do not change this: from here
np.random.seed(0)

M = 1000
N = 1000

A = np.random.rand(M, N)
b = np.random.rand(M)


# do not change this: till here


def constraint_indicator_func(x):
    """ The constraint per the assignment seems to be abs(x_i) <= 1"""
    # DONE: Construct a indicator function which returns True if x belongs to constraint set else False
    return abs(x) <= 1


def objective_value(x):
    # DONE: Calculated objective value if x belongs to constraint set else set to infinity
    x_filter = x.copy()
    x_filter[not constraint_indicator_func(x_filter)] = np.inf
    norm_param = A * x_filter - b
    return (1 / 2) * (np.linalg.norm(norm_param, ord=2) ** 2)


def gradient_of_objective(x):
    ...  # TODO: Calculate the gradient of the smooth part of the objective


def linear_minimization_oracle(x, a):
    # DONE: Linear minimization problem over the constraint set
    # argmin_{x in C} (<x,a>) for some a
    # and C is the constraint set given in the question
    # return argmin here
    x = -1*np.sign(a)
    return x


# Conditional Gradient Method
x_init = np.random.rand(N)

# do not change this
max_iter = 1000

x_k = x_init.copy()

# Taping Objective Values and Time
obj_vals = [objective_value(x_k)]
time_vals = [1e-2]  # adding 1e-2 just to give a starting time

for iter in range(max_iter):
    ...  # TODO: Complete Conditional Gradient Descent iteration and save the objective value into obj_vals after the update
    # TODO: Also track time in each iteration and store the time take per iteration into time_vals


# Projected Gradient Method

def projection(x):
    ...  # TODO: Calculate the projection onto the constraint set


x_k_1 = x_init.copy()

# Taping Objective Values and Time
obj_vals_1 = [objective_value(x_k_1)]
time_vals_1 = [1e-2]

# TODO: Calculate the Lipschitz constant and save into Lips_val
Lips_val = ...

# do not change this
tau = (2 / (Lips_val)) * 0.9

for iter in range(max_iter):
    ...  # TODO: Complete Projected Gradient Descent iteration and save the objective value into obj_vals_1 after the update
    # TODO: Also track time in each iteration and store the time take per iteration into time_vals_1

fig1 = plt.figure()

# Denote CGM for Conditional Gradient Method 
# and PGM for Proximal Gradient Method 
# (here PGM is equivalent to Projected Gradient Descent)

# TODO: plot log plot of objective values of CGM and PGM with respect to iterations

# TODO: Use appropriate labels and legend

# TODO: Save figure with file name 'objective_vs_iterations.png'


fig2 = plt.figure()

# TODO: plot cumulative sum of time values of CGM and PGM with respect to iterations

# TODO: Use appropriate labels and legend

# TODO: Save figure with file name 'time_vs_iterations.png'


fig3 = plt.figure()

# TODO: plot log log plot of objective values of CGM and PGM with respect to cumulative sum of time values

# TODO: Use appropriate labels and legend

# TODO: Save figure with file name 'objective_vs_time.png'
