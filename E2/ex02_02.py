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


# Alternating projection method
def POCS(x0, tol, maxiter, check):
    
    # initialization
    x = x0.copy();
    y = x.copy();

    # taping
    seq = [x0];

    for iter in np.arange(0, maxiter):
       
        ###########################################################################
        ### TODO: Perform the update step of the Alternating Projection Method. ###
        # Denote the points on $C$ by $y$ and the points on $D$ by $x$. Append
        # $y$ and $x$ to the list 'seq'. 
        # Implement an appropriate breaking condition. Define an appropriate
        # residual and break the loop if res<tol.
	#
	# WRITE YOUR SOLUTION HERE
	#
        ###########################################################################

        # provide some information
        if iter % check == 0:
            print ('Iter: %d, res: %f' % (iter, res));
    
    return seq;


# Dykstra's projection method
def Dykstra(x0, tol, maxiter, check):
    
    # initialization
    x = x0.copy();
    y = x.copy();
    p = np.zeros(x.shape);
    q = np.zeros(x.shape);

    # taping
    seq = [x0];

    for iter in np.arange(0, maxiter):
        
        ###########################################################################
        ### TODO: Perform the update step of the Dykstra's  Projection Method.  ###
        # Denote the points on $C$ by $y$ and the points on $D$ by $x$. Append
        # $y$ and $x$ to the list 'seq'. 
        # Implement an appropriate breaking condition. Define an appropriate
        # residual and break the loop if res<tol.
	#
	# WRITE YOUR SOLUTION HERE
	#
        ###########################################################################

        # provide some information
        if iter % check == 0:
            print ('Iter: %d, res: %f' % (iter, res));
    
    return seq;


### Run the algorithms ###
x0 = np.array([1.25,-2]);
tol = 1e-10;
maxiter = 100;
check = 10;

###################################################################################
### TODO: Find an initialization x0 for which POCS and Dykstra's Algorithm converge 
# to (clearly) different points. [Just redefine x0 here!] 
#
# WRITE YOUR SOLUTION HERE
#
###################################################################################


# run the POCS algorithm (Alternating Projection Method)
seq_POCS = POCS(x0, tol, maxiter, check);
arr_POCS = np.vstack(seq_POCS).T;

# run Dykstra Projection Algorithm
seq_Dykstra = Dykstra(x0, tol, maxiter, check);
arr_Dykstra = np.vstack(seq_Dykstra).T;


### visualize the result:
fig = plt.figure();
plt.subplot(1,2,1);
plt.axis('equal')
# draw a part of the set D
plt.plot([2,-1],[-1,2],color=(0,0,0,1),linewidth=2);
# draw the boundary of the set C
an = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(an), np.sin(an),color=(0,0,0,1),linewidth=1);
# visualize iterates of POCS
plt.plot(arr_POCS[0,:], arr_POCS[1,:], '-x', color=(1,0,0,1),linewidth=1,markersize=4);
plt.plot(arr_POCS[0,-1], arr_POCS[1,-1], '-x', color=(1,0,0,1),linewidth=2,markersize=8);
plt.title("POCS");

plt.subplot(1,2,2);
plt.axis('equal')
# draw a part of the set D
plt.plot([2,-1],[-1,2],color=(0,0,0,1),linewidth=2);
# draw the boundary of the set C
an = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(an), np.sin(an),color=(0,0,0,1),linewidth=1);
# visualize iterates of Dykstra's Algorithm
plt.plot(arr_Dykstra[0,:], arr_Dykstra[1,:], '-*', color=(0,0,1,1),linewidth=1,markersize=4);
plt.plot(arr_Dykstra[0,-1], arr_Dykstra[1,-1], '*', color=(0,0,1,1),linewidth=2,markersize=8);
plt.title("Dykstra");
# show result
plt.show();





