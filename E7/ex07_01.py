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
f = t ** 2 - 0.5 * t
s = np.zeros(n - 1)
s_before = -float('Inf')  # s_before for s_{-}
s_after = float('Inf')  # s_after for s_{+}

# DONE: Compute slopes
for i in np.arange(0, n - 1):
    f_diff = f[i] - f[i + 1]
    t_diff = t[i] - t[i + 1]
    s[i] = f_diff / t_diff

# DONE: CALCULATE THE SUBDIFFERENTIAL OVER t
sub_t, subval = np.zeros(n*2), np.zeros(n*2)

for i in range(n):
    first, second = None, None

    if i == 0:
        first = s_before
        second = s[0]

    elif i >= len(s):
        first = s[-1]
        second = s_after

    else:
        first = s[i-1]
        second = s[i]

    # We have each value of t twice for plotting the vertical line
    sub_t[i*2] = t[i]
    sub_t[i*2 + 1] = t[i]

    subval[i*2] = first
    subval[i*2 + 1] = second

subval = np.array(subval).flatten()
sub_t = np.array(sub_t).flatten()

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
# DONE: SET APPROPRIATE TITLE
ax1.set_title("The subdifferential of f with regard to breakpoints t")


# DONE: PLOT THE OBTAINED SUBDIFFERENTIAL VS t.
ax1.plot(sub_t, subval)

ax1.grid(True, which='both')
plt.tight_layout()
ax1.set_xlim(-5, 5)  # restrict the plot to -5 till 5 interval on x axis
# DONE: SAVE THE FIGURE AS 'subdifferential_plot.png'
fig.savefig('subdifferential_plot.png')

plt.show()
