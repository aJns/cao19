import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

### load data
# A: data matrix of size M x N
# (nx,ny): image dimensions (number of pixels in horizontal and vertical direction
A, N, M, nx, ny = np.load('data/data.npy')


### auxilliary function
def convert2image(A, t):
    return A[:, t].reshape(ny, nx)


### optimization problem:
#
# min_{X,Y} mu*||Y||_* + rho*||X||_1 + 0.5*||X+Y-A||_F^2
#


### model parameters
###############################################################################
# TODO: 
# - select suitable parameters $mu$ and $rho$ of the optimization model ###

###############################################################################

### Proximal Gradient Descent
maxiter = 10000
check = 10  # print the objective value and the rank of $Y$ after $check$ iterations
tol = 1e-6  # break iterations if the residual $res$ drops below $tol$

# initialize 
###############################################################################
# TODO: 
# - find a suitable intialization for the unknowns $X$ and $Y$
# - compute the Lipschitz constant $L$ of the gradient of the smooth part of the objective 
# - set the step size parameter $tau$ such that the algorithm converges

###############################################################################


for iter in np.arange(0, maxiter):

    ###############################################################################
    # TODO: implement the Proximal Gradient Step 

    ###############################################################################

    ###############################################################################
    # TODO: 
    # - compute the objective value $val$ 
    # - compute the current rank $rk$ of $Y$

    ###############################################################################

    ###############################################################################
    # TODO: 
    # - implement a suitable stopping criterion $res$<$tol$ for the algorithm

    if res < tol:
        break
    ###############################################################################

    if iter % check == 0:
        print('iter: %d, val: %f, rank: %d' % (iter, val, rk))

# visualize for every 10th image the outliers and the low-rank part
fig = plt.figure()
NN = N / 10.0
gx = int(np.ceil(np.sqrt(NN)))
for i in np.arange(0, NN):
    plt.subplot(gx, gx, i + 1)
    img = convert2image(X, int(10 * i))
    plt.imshow(img)

fig2 = plt.figure()
gx = int(np.ceil(np.sqrt(NN)))
for i in np.arange(0, NN):
    plt.subplot(gx, gx, i + 1)
    img = convert2image(Y, int(10 * i))
    plt.imshow(img)

plt.show()

# write 3 outlier masks
img = convert2image(X, 10)
misc.imsave('outlier0010.png', img)
img = convert2image(X, 50)
misc.imsave('outlier0050.png', img)
img = convert2image(X, 99)
misc.imsave('outlier0099.png', img)
img = convert2image(Y, 50)
misc.imsave('background0050.png', img)
