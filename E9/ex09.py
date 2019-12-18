import matplotlib.pyplot as plt
import numpy as np

### load data
# A: data matrix of size M x N
# (nx,ny): image dimensions (number of pixels in horizontal and vertical direction
A, N, M, nx, ny = np.load('data/data.npy', allow_pickle=True)


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

# Let's start out with 1s
mu = 1
rho = 1

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

array_shape = M, N

X = np.ones(array_shape)
Y = np.ones(array_shape)

# YOLOOO
lipschitz_constant = 1
tau = 1


###############################################################################

def objective_function(X, Y, A, array_shape, mu, rho):
    h_term = X + Y - A
    h = 0.5 * (np.linalg.norm(h_term, ord='fro') ** 2)
    g = rho * np.linalg.norm(X, ord=1) + mu * np.linalg.norm(Y, ord='nuc')

    return h + g


def h_gradient_descent(X, Y):
    h_gradient = 2 * X + 2 * Y
    X_term = X - tau * h_gradient
    Y_term = Y - tau * h_gradient
    return X_term, Y_term


def g_proximal_map(X, Y):
    # the proximal map for X is prox_{rho||X||_1} + mu*||Y||_*
    # the proximal map for Y is prox_{mu||Y||_*} + rho*||X||_1

    def X_prox(X):
        prox_param = tau
        return np.maximum(0, np.abs(X) - prox_param) * np.sign(X)

    def Y_prox(Y):
        prox_param = tau

        def st(A):
            A[prox_param < A] -= prox_param
            A[A < -prox_param] += prox_param
            A[(-prox_param <= A) & (A <= prox_param)] = 0

            return A

        u, s, vh = np.linalg.svd(Y)
        S = st(s)
        s_shape = S.shape[0]
        return np.dot(u[:, :s_shape] * S, vh)

    X_next = X_prox(X) + mu * np.linalg.norm(Y, ord='nuc')
    Y_next = Y_prox(Y) + rho * np.linalg.norm(X, ord=1)

    return X_next, Y_next


prev_val = np.Inf

for iter in np.arange(0, maxiter):

    ###############################################################################
    # TODO: implement the Proximal Gradient Step

    X, Y = h_gradient_descent(X, Y)
    X, Y = g_proximal_map(X, Y)

    ###############################################################################

    ###############################################################################
    # TODO: 
    # - compute the objective value $val$ 
    # - compute the current rank $rk$ of $Y$

    val = objective_function(X, Y, A, array_shape, mu, rho)
    rank = np.linalg.matrix_rank(Y)

    ###############################################################################

    ###############################################################################
    # TODO: 
    # - implement a suitable stopping criterion $res$<$tol$ for the algorithm

    res = np.abs(prev_val - val)

    if res < tol:
        break
    ###############################################################################

    if iter % check == 0:
        print('iter: %d, val: %f, rank: %d' % (iter, val, rank))

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
plt.imsave('outlier0010.png', img)
img = convert2image(X, 50)
plt.imsave('outlier0050.png', img)
img = convert2image(X, 99)
plt.imsave('outlier0099.png', img)
img = convert2image(Y, 50)
plt.imsave('background0050.png', img)
