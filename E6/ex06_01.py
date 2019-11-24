import numpy as np
import matplotlib.pyplot as plt

# Let $f$ be a continuous piecewise linear function. We represent $f$ by a sequences
# (t_0, ..., t_{n-1}) of breakpoints
# (f_0, ..., f_{n-1}) values of f at the breakpoints
# (s_0, ..., s_{n-1}) slopes of f, s_i is the slope on (t_{i},t_{i+1}) for 0<i<n
#                          and s_{-1} is the slope on (-\infty,t_0),
#                          and s_{n-1} the slope on (t_{n-1},+\infty)
# Note that this information is redundant.
#

n = 5

# breakpoints
t = np.linspace(-5, 5, n)

# function evaluations at breakpoints
f = t ** 2 - 0.5 * t

s_before = -np.inf  # s_before for s_{-1}

# comprises of s[0] till s[n-1] as per the above description
s = np.zeros(n - 1)  # tentative slopes, see TODO below

s_after = np.inf  # s_after for s_{n-1}

# Note that the domain is only from -5 till 5


# The 's' array contains only tentative slopes, 
# you need to calculate the actual slopes below 
# using the function values at breakpoints.
for i in np.arange(n):
    s[i - 1] = (f[i] - f[i - 1]) / (t[i] - t[i - 1])

print(f)
print(t)
print(s)

# TODO: remove these
plt.plot(t, f)
plt.show()


def conjugate_function(y, func_vals, breakpoints):
    # y is the point of evaluation
    # func_vals : Function values
    # breakpoints : original breakpoints
    # TODO: Complete the conjugate function using function values and breakpoints
    return


n_2 = 50
t_2 = np.linspace(-10, 10, n_2)

f_2 = None  # TODO: Evaluate conjugate_function here over t_2

# Starting the plot
# We want to compare original function and conjugate function side by side.
fig = plt.figure()

# You may use x axis limits from -10 till 10.

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('Original function')
# TODO: Plot the original function with f on y axis and t on x axis.
# Plot only on the domain of the function.
# You may skip plotting infinities here.
# Also set the grid of both the axes to 'True'. 

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('Conjugate function')
# TODO: Plot the conjugate function with f_2 on y axis and t_2 on x axis.
# Also set the grid of both the axes to 'True'.


# TODO: Save the figure as 'original_and_conjugate_functions.png'
