"""
    This coding exercise involves checking the convexity of a piecewise linear function. 
    
    You task is to fill in the function "convex_check".

    In the end, when the file is run with "python3 ex04_01.py" command, it should display the total number of convex functions.
"""

# basic numpy import
import numpy as np

import time

import matplotlib.pyplot as plt

# random seed fix, do not change this
np.random.seed(0)

initial_func_val = 0  # initial function value which is f(0)=0

# this creates an array of slopes to be checked
slopes_array = np.random.randint(10, size=(100, 5))

# each row within slopes array represents the sequence of slopes m_i's 
# m_i represents the slope within the interval (t_i,t_{i+1})
# for example: m_1 = slopes[0] is the slope within [a,t_1]


# List of 5 Break points
# a = t_1 = 0, t_2 = 20, t_3 = 40, t_4 = 60,  b = t_5 = 100
# we collect all the points into the following list
break_points = [0, 20, 40, 60, 80, 100]


# Helpful for visualization
def plot_function(slopes, break_points):
    x = np.array(break_points)
    y = np.array([x[0]])

    for i in range(len(slopes)):
        new_y = y[i] + slopes[i] * (x[i + 1] - x[i])
        y = np.append(y, new_y)

    plt.plot(x, y)
    plt.show()


def convex_check(slopes, break_points):
    """Checks if the function is convex or not.

    Arguments:
        slopes {np.array} -- List of Slopes
        break_points {np.array} -- List of Breakpoints
    """

    convexity = True

    # If the slope is smaller than the last one, the function is non-convex
    prev_slope = slopes[0]
    for slope in slopes[1:]:
        if slope < prev_slope:
            convexity = False
            break
        else:
            prev_slope = slope

    return convexity


convex_func_count = 0
for slopes in slopes_array:
    if convex_check(slopes, break_points):
        convex_func_count += 1
    else:
        pass
print('Number of convex functions: ', convex_func_count)
