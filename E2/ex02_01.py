import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def my_func(x, y):
    return (1 - x * y) ** 2


# creating the grid using np.linspace for plotting
# 100 samples are enough
sample_count = 100
start, stop = -10, 10

x = np.linspace(start, stop, sample_count)
y = np.linspace(start, stop, sample_count)

# create meshgrid 
X, Y = np.meshgrid(x, y)

# Create Z by using my_func over the grid
Z = my_func(X, Y)

# Start create the figure and the axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Now create surface plot
surf = ax.plot_surface(X, Y, Z)

# Create x,y,z labels
ax.set_title("Surface plot of f(x,y) = (1 - xy)^2")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Creating a legend doesn't seem useful as there is only one surface

# Create figures in PDF and PNG format.
try:
    os.mkdir("output")
except FileExistsError:
    pass

plt.savefig("output/surface_plot.pdf")
plt.savefig("output/surface_plot.png")

plt.show()
