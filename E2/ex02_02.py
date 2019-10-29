import numpy as np
import matplotlib.pyplot as plt


################################################################################
# Find the projection of $x$ onto the unit simplex in $\R^2$:                  #
#                                                                              #
#       S = {x=(x_1,x_2) | x_1 + x_2 = 1 and x_1 >=0 and x_2 >= 0}             #
#                                                                              #
# We seek for an iterative solution using the alternating projection method    #
# or Dykstra's projection algorithm.                                           #
#                                                                              #
# The set $S$ can be described as the intersection of the following sets $C$   #
# and $D$:                                                                     #
#                                                                              #
#       C = {x=(x_1,x_2) | x_1^2 + x_2^2 <= 1}                                 #
#       D = {x=(x_1,x_2) | x_1 + x_2 = 1}                                      #
#                                                                              #
################################################################################

def projD(x0):
    # line equation
    a, b, c = 1, 1, -1
    x = np.zeros(2)

    # point on the line ax + by + c = 0 with the shortest distance ie the projection
    num = b*(b*x0[0] - a*x0[1]) - a*c
    denom = a**2 + b**2
    x[0] = num / denom

    num = a*(-b*x0[0] + a*x0[1]) - b*c
    x[1] = num / denom

    return x


def projC(y0):
    # Projection onto the euclidean norm ball
    # Check example A.5
    # quickly:
    #   projC(x) = x / max(1, l2norm(x))

    # numpy probably has an l2 norm function but lets be cool and do it ourselves
    l2norm = np.sqrt(sum([y**2 for y in y0]))

    return y0 / max(1, l2norm)

def distance(x1, x2):
    return np.sqrt(sum([x**2 for x in x1-x2]))


def residual(current, previous):
    return distance(current, previous)


# Alternating projection method
def POCS(x0, tol, maxiter, check):
    # initialization
    x = x0.copy()
    y = x.copy()
    res = tol

    iter_count = 0

    # taping
    seq = [x0]

    for iter in np.arange(0, maxiter):

        iter_count = iter

        ###########################################################################
        ### DONE: Perform the update step of the Alternating Projection Method. ###
        # Denote the points on $C$ by $y$ and the points on $D$ by $x$. Append
        # $y$ and $x$ to the list 'seq'. 
        # Implement an appropriate breaking condition. Define an appropriate
        # residual and break the loop if res<tol.

        x = projC(seq[-1])
        y = projD(x)

        seq.append(x)
        seq.append(y)

        res = residual(seq[-1], seq[-2])

        if res < tol:
            print("Stopped iteration since res < tol")
            print("\tres: {}, tol: {}".format(res, tol))
            break

        ###########################################################################

        # provide some information
        if iter % check == 0:
            print('Iter: %d, res: %f' % (iter, res))

    print("Ran for {} iterations".format(iter_count))

    return seq


# Dykstra's projection method
def Dykstra(x0, tol, maxiter, check):
    # initialization
    x = x0.copy()
    y = x.copy()
    p = np.zeros(x.shape)
    q = np.zeros(x.shape)

    iter_count = 0

    # taping
    seq = [x0]

    for iter in np.arange(0, maxiter):

        iter_count = iter

        ###########################################################################
        ### DONE: Perform the update step of the Dykstra's  Projection Method.  ###
        # Denote the points on $C$ by $y$ and the points on $D$ by $x$. Append
        # $y$ and $x$ to the list 'seq'. 
        # Implement an appropriate breaking condition. Define an appropriate
        # residual and break the loop if res<tol.

        y = projC(x + p)

        p = x + p - y

        x = projD(y + q)

        q = y + q - x

        seq.append(y)
        seq.append(x)

        res = residual(seq[-1], seq[-2])

        if res < tol:
            print("Stopped iteration since res < tol")
            print("\tres: {}, tol: {}".format(res, tol))
            break

        ###########################################################################

        # provide some information
        if iter % check == 0:
            print('Iter: %d, res: %f' % (iter, res))

    print("Ran for {} iterations".format(iter_count))

    return seq


### Run the algorithms ###
x0 = np.array([1.25, -2])
tol = 1e-10
maxiter = 100
check = 10

###################################################################################
### DONE: Find an initialization x0 for which POCS and Dykstra's Algorithm converge
# to (clearly) different points. [Just redefine x0 here!] 

x0 = np.array([0.5, 1.5])

###################################################################################

# %%

# run the POCS algorithm (Alternating Projection Method)
seq_POCS = POCS(x0, tol, maxiter, check)
arr_POCS = np.vstack(seq_POCS).T

# %%

# run Dykstra Projection Algorithm
seq_Dykstra = Dykstra(x0, tol, maxiter, check)
arr_Dykstra = np.vstack(seq_Dykstra).T

# %%

# Some metrics
POCS_distance = distance(seq_POCS[0], seq_POCS[-1])
Dykstra_distance = distance(seq_Dykstra[0], seq_Dykstra[-1])

print("POCS distance to result:", POCS_distance)
print("Dykstra distance to result:", Dykstra_distance)

### visualize the result:
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.axis('equal')
# draw a part of the set D
plt.plot([2, -1], [-1, 2], color=(0, 0, 0, 1), linewidth=2)
# draw the boundary of the set C
an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an), color=(0, 0, 0, 1), linewidth=1)
# visualize iterates of POCS
plt.plot(arr_POCS[0, :], arr_POCS[1, :], '-x', color=(1, 0, 0, 1), linewidth=1, markersize=4)
plt.plot(arr_POCS[0, -1], arr_POCS[1, -1], '-x', color=(1, 0, 0, 1), linewidth=2, markersize=8)
plt.title("POCS")

# %%

plt.subplot(1, 2, 2)
plt.axis('equal')
# draw a part of the set D
plt.plot([2, -1], [-1, 2], color=(0, 0, 0, 1), linewidth=2)
# draw the boundary of the set C
an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an), color=(0, 0, 0, 1), linewidth=1)
# visualize iterates of Dykstra's Algorithm
plt.plot(arr_Dykstra[0, :], arr_Dykstra[1, :], '-*', color=(0, 0, 1, 1), linewidth=1, markersize=4)
plt.plot(arr_Dykstra[0, -1], arr_Dykstra[1, -1], '*', color=(0, 0, 1, 1), linewidth=2, markersize=8)
plt.title("Dykstra")
# show result
plt.show()
