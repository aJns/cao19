
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

# DO NOT CHANGE THIS SECTION
n = 5
t = np.linspace(-5, 5, n)
f = t**2 - 0.5*t
s = np.zeros(n-1)
s_before = -float('Inf')         # s_before for s_{-}
s_after = float('Inf')          # s_after for s_{+}

# TODO: Compute slopes
for i in np.arange(0,n-1):
    pass
    
# TODO: CALCULATE THE SUBDIFFERENTIAL OVER t


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
# TODO: SET APPROPRIATE TITLE


# TODO: PLOT THE OBTAINED SUBDIFFERENTIAL VS t.


ax1.grid(True, which='both')
plt.tight_layout()
ax1.set_xlim(-5,5) # restrict the plot to -5 till 5 interval on x axis
# TODO: SAVE THE FIGURE AS 'subdifferential_plot.png'
