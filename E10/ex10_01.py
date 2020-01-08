import time
import numpy as np

import matplotlib.pyplot as plt
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
img_2d_shape = (ny, nx)
print("image dim: ", np.shape(img))

### constuct the cost matrix
fg_color = np.array([0, 0, 1])
bg_color = np.array([0, 0, 0])

rho = 0.1  # regularization weight
cost = np.zeros((ny, nx))
ones = np.ones((ny, nx))
for i in range(3):
    cost = cost + (img[:, :, i] - fg_color[i] * ones) ** 2
    cost = cost - (img[:, :, i] - bg_color[i] * ones) ** 2
c = np.reshape(rho * cost, nx * ny)  # vectorize cost image into cost vector

maxiter = 500
check = 10

tau0 = 2.0
tau = tau0 / 8.0
val_ = 1e15

K = make_derivatives2D(ny, nx)
u_kp1 = np.ones(nx * ny)
u_k = u_kp1.copy()


def objective_function(u):
    inner_product = np.dot(c, u)
    norm = np.linalg.norm(K * u, ord=1)
    return inner_product * norm


def calc_subgradient(u):
    return np.gradient(K*u)[:ny*nx]


def project(descent_step):
    return descent_step - c


cost_values = []

plt.figure(1)
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

# fig1 = plt.figure(1);
# plt.show(block=False);
for iter in np.arange(0, maxiter):

    # TODO:  Complete the following todos
    u_k = u_kp1

    # TODO: Change tau appropriately for subgradient descent

    # TODO: compute subgradient
    subgradient = calc_subgradient(u_k)

    # DONE: compute objective value at u_k
    val = objective_function(u_k)
    cost_values.append(val)

    # TODO: update step
    descent_step = u_k - tau * subgradient

    # TODO: projection step
    u_kp1 = project(descent_step)

    # print information
    if iter % check == 0:
        ## show current result
        # TODO: display current result after each 'check' iteration
        # DONE: Create 'img_u' by reshaping u_kp1 appropriately
        img_u = np.reshape(u_kp1, img_2d_shape)

        # DONE: Create a segmentation colored rgb image 'col_img' where if an element
        # of u_kp1 is >0 then set the class to 1 else 0
        col_img2d = img_u.copy()
        col_img2d[col_img2d > 0] = 1
        col_img2d[col_img2d <= 0] = 0

        # DONE: And, then all pixels belonging to class 0 should be given blue color
        # and class 1 is denoted by red color (foreground red, background blue)
        # Note: First reshape u_kp1 appropriately.
        col_img_red = col_img2d.copy()
        col_img_blue = np.logical_not(col_img2d.copy())
        col_img_green = np.zeros(img_2d_shape)
        col_img = np.stack([col_img_red, col_img_green, col_img_blue], axis=-1)

        # DONE: Create subplots where you show cost, img_u and col_img side by side.
        # Maybe gotta clear these each iteration?
        ax1.plot(cost_values)
        ax2.imshow(img_u)
        ax3.imshow(col_img)

        # TODO: add titles and make plots nicer (add color bar if required)

        plt.pause(0.01)  # to visualize the changes
        # plt.pause(10)  # to visualize the changes

        print("iter: %d, tau: %f, val: %f" % (iter, tau, val))
plt.close()

### evaluation (apply classifier to all pixels)

# TODO: Create 'img_u' by reshaping u_kp1 appropriately
# TODO: Create 'col_img' like before.

# TODO construct overlay image 'convex combination of original image and 
# segmentation (col_img)'; You may set the weight for 'col_img' to 0.45.

overlay = ...  # TODO

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
