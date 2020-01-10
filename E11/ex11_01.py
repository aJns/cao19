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
    return np.all(np.abs(x) <= 1)


def objective_value(x):
    # DONE: Calculated objective value if x belongs to constraint set else set to infinity
    retval = np.inf
    if constraint_indicator_func(x):
        norm_param = A * x - b
        retval = (1 / 2) * np.sum(np.power(norm_param, 2))
    return retval


def gradient_of_objective(x):
    # DONE: Calculate the gradient of the smooth part of the objective
    return (A*x-b)*A


def linear_minimization_oracle(x, a):
    # DONE: Linear minimization problem over the constraint set
    # argmin_{x in C} (<x,a>) for some a
    # and C is the constraint set given in the question
    # return argmin here
    x = -1 * np.sign(a)
    return x


# Conditional Gradient Method
x_init = np.random.rand(N)

# do not change this
max_iter = 1000

x_k = x_init.copy()

# Taping Objective Values and Time
obj_vals = [objective_value(x_k)]
time_vals = [1e-2]  # adding 1e-2 just to give a starting time

start_time = time.time()

for iter in range(max_iter):
    # DONE: Complete Conditional Gradient Descent iteration and save the objective value into obj_vals after the update
    y_k = linear_minimization_oracle(None, A)
    gamma_k = 2 / (iter + 2)
    x_k = y_k * gamma_k + (1 - gamma_k) * x_k
    obj_vals.append(objective_value(x_k))

    # DONE: Also track time in each iteration and store the time take per iteration into time_vals
    time_vals.append(time.time() - start_time)

    print("Conditional Gradient Descent iteration: {}/{}".format(iter, max_iter))


# Projected Gradient Method

def projection(x):
    """Since we're dealing with a simple |x_i| for i=1..N, we can just clip all elements outside our constraint"""
    # DONE: Calculate the projection onto the constraint set
    x[x > 1] = 1
    x[x < -1] = -1
    return x


x_k_1 = x_init.copy()

# Taping Objective Values and Time
obj_vals_1 = [objective_value(x_k_1)]
time_vals_1 = [1e-2]

# DONE: Calculate the Lipschitz constant and save into Lips_val
Lips_val = np.abs(A * A)  # Absolute of the derivative of the gradient of h(x); Should be the maximum rate of change

# do not change this
tau = (2 / (Lips_val)) * 0.9

start_time = time.time()

for iter in range(max_iter):
    # DONE: Complete Projected Gradient Descent iteration and save the objective value into obj_vals_1 after the update
    to_project = x_k_1 - tau * gradient_of_objective(x_k_1)
    x_k_1 = projection(to_project)
    obj_vals_1.append(objective_value(x_k_1))

    # DONE: Also track time in each iteration and store the time take per iteration into time_vals_1
    time_vals_1.append(time.time() - start_time)

    print("Projected Gradient Descent iteration: {}/{}".format(iter, max_iter))

fig1 = plt.figure()

# Denote CGM for Conditional Gradient Method 
# and PGM for Proximal Gradient Method 
# (here PGM is equivalent to Projected Gradient Descent)

# DONE: plot log log plot of objective values of CGM and PGM with respect to iterations
plt.loglog(obj_vals, label="CGM")
plt.loglog(obj_vals_1, label="PGM")

# DONE: Use appropriate labels and legend
ax = plt.gca()
ax.set_xlabel("Iterations")
ax.set_ylabel("Objective values")

plt.legend(loc="upper left")


# DONE: Save figure with file name 'objective_vs_iterations.png'
fig1.savefig("objective_vs_iterations.png")


fig2 = plt.figure()

# DONE: plot cumulative sum of time values of CGM and PGM with respect to iterations
plt.plot(time_vals, label="CGM")
plt.plot(time_vals_1, label="PGM")

# DONE: Use appropriate labels and legend
ax = plt.gca()
ax.set_xlabel("Iterations")
ax.set_ylabel("Total time taken")

plt.legend(loc="upper left")

# DONE: Save figure with file name 'time_vs_iterations.png'
fig2.savefig("time_vs_iterations.png")


fig3 = plt.figure()

# DONE: plot log log plot of objective values of CGM and PGM with respect to cumulative sum of time values
plt.loglog(time_vals, obj_vals, label="CGM")
plt.loglog(time_vals_1, obj_vals_1, label="PGM")

# DONE: Use appropriate labels and legend
ax = plt.gca()
ax.set_xlabel("Total time taken")
ax.set_ylabel("Objective values")

plt.legend(loc="upper left")

# DONE: Save figure with file name 'objective_vs_time.png'
fig3.savefig("objective_vs_time.png")

plt.show()
