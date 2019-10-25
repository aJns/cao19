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

# Create figures in PDF and PNG format.
plt.show()
