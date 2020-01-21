import time
import numpy as np

import matplotlib.pyplot as plt
import myimgtools
from myimgtools import make_derivatives2D

### Solve c ############################################
#
# min_{u \in [-1,1]^N} <c,u> + ||K*u||_1
#
# where
#
#    u       relaxed binary segmentation (vectorized image of dimension N)
#    c       cost matrix (vectorized image of dimension N)
#    K       discrete x- and y-derivative operator (2N x N - matrix)
#
# Note that the regularization weight can be absorbed in the cost matrix c
#
###############################################################################

### load data
filename_image = 'data/flower.png'
img = plt.imread(filename_image)
(ny, nx, nc) = np.shape(img)
print("image dim: ", np.shape(img))

### constuct the cost matrix
fg_color = np.array([0, 0, 1])
bg_color = np.array([0, 0, 0])

rho = 0.1  # regularization weight
cost = np.zeros((ny, nx))
ones = np.ones((ny, nx))
for i in np.arange(0, 3):
    cost = cost + (img[:, :, i] - fg_color[i]*ones)**2
    cost = cost - (img[:, :, i] - bg_color[i]*ones)**2
c = np.reshape(rho*cost, nx*ny)  # vectorize cost image into cost vector

K = make_derivatives2D(ny, nx)
lam = 0.1
c = c/lam # rescaling c


# initialization
x = np.zeros(nx*ny)
y = np.zeros(nx*ny*2)
x_hat = x.copy() # copying x for later use

# algorithm parametes
maxiter = 500
check = 10

L = np.sqrt(8.0)  

# Step-sizes in Primal-Dual Hybrid Gradient Algorithm
tau = 10        
sigma = 0.9/(L*L*tau)   
# You are free to explore your own choice for tau and sigma

def objective_val(x):
    return c.dot(x) + np.linalg.norm(K @ u, ord=1)

initial_objective = objective_val(x)    # DONE: Calculate the objective at the starting point
objective_vals = [initial_objective]


for iter in np.arange(0, maxiter):

    # Complete the following todos

    # TODO: Complete the Primal-Dual Hybrid Gradient Algorithm
    # Denote x as the primal variable
    # and y to be the dual variable
    # Clearly distinguish the primal update and the dual update
    

    # DONE: Compute Objective value at x and add it to 'objective_vals' list
    val = objective_val(x)
    objective_vals.append(val)

    # print information
    if iter % check == 0:
        ## -- Same as Exercise 4 of Sheet 10 -- FROM HERE
        ## show current result
        # TODO: display current result after each 'check' iteration
        plt.figure(1)
        # TODO: Create 'img_u' by reshaping x appropriately

        # TODO: Create a segmentation colored rgb image 'col_img' where if an element
        # of x is >0 then set the class to 1 else 0
        # TODO: And, then all pixels belonging to class 0 should be given blue color
        # and class 1 is denoted by red color (foreground red, background blue)
        # Note: First reshape x appropriately.

        # TODO: Create subplots where you show cost, img_u and col_img side by side.
        # TODO: add titles and make plots nicer (add color bar if required)

        plt.pause(0.01)  # to visualize the changes

        print("iter: %d, tau: %f, val: %f" % (iter, tau, val))
plt.close()

### evaluation (apply classifier to all pixels)

# TODO: Create 'img_u' by reshaping x appropriately
# TODO: Create 'col_img' like before.

# TODO construct overlay image 'convex combination of original image and
# segmentation (col_img)'; You may set the weight for 'col_img' to 0.45.

overlay =  # TODO

# TODO: add titles and make plots nicer (other color bar)

plt.figure()
# TODO: Create subplots where you show cost, img_u and overlay side by side.
# TODO: add titles and make plots nicer (add color bar if required)
plt.show()


# TODO: save the variable 'col_img' to 'segmentation.png'

# TODO: save the variable 'overlay' to 'overlay.png'

# TODO: Consider following naive thresholding based segmentation.
# Pixels with negative values in img_c  are set to class 0
# Pixels with positive values in img_c  are set to class 1

# The create naive_img to represent the segmentation obtained naively.
# And, then all pixels belonging to class 0 should be given blue color
# and class 1 is denoted by red color (foreground red, background blue)

# Start naive thresholding implementation
naive_img = np.zeros((ny, nx, 3))  # visualizes the naive segmentation

# TODO: Save the segmentation obtained to the variable 'naive_img'


# TODO: save the variable 'naive_img' to 'naive_img.png'
## -- Same as Exercise 4 of Sheet 10 -- TILL HERE

## -- Plotting objective vs iterations

#  Reuse the code of Exercise 4 of Sheet 10 to track objective values of
#  Projected Subgradient Method (PSM)

# Denote Primal Dual Hybrid Gradient Method  as PDHG.

# TODO: Complete the following figure to compare the performance of PDHG and PSM 
# on objective over iterations.

fig = plt.figure()

# TODO: plot log log plot of objective values  with respect to iterations

# TODO: Use appropriate labels and legend

# TODO: Save figure with file name 'objective_vs_iterations.png'


