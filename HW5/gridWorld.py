import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


##
def genGridWorld():
    O = -1e5  # Dangerous places to avoid
    D = 35  # Dirt
    W = -100  # Water
    C = -3000  # Cat
    T = 1000  # Toy
    grid_list = {0: '', O: 'O', D: 'D', W: 'W', C: 'C', T: 'T'}
    grid_world = np.array([[0, O, O, 0, 0, O, O, 0, 0, 0],
                           [0, 0, 0, 0, D, O, 0, 0, D, 0],
                           [0, D, 0, 0, 0, O, 0, 0, O, 0],
                           [O, O, O, O, 0, O, 0, O, O, O],
                           [D, 0, 0, D, 0, O, T, D, 0, 0],
                           [0, O, D, D, 0, O, W, 0, 0, 0],
                           [W, O, 0, O, 0, O, D, O, O, 0],
                           [W, 0, 0, O, D, 0, 0, O, D, 0],
                           [0, 0, 0, D, C, O, 0, 0, D, 0]])
    return grid_world, grid_list


##
def showWorld(grid_world, tlt):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(tlt)
    ax.set_xticks(np.arange(0.5, 10.5, 1))
    ax.set_yticks(np.arange(0.5, 9.5, 1))
    ax.grid(color='b', linestyle='-', linewidth=1)
    ax.imshow(grid_world, interpolation='nearest', cmap='copper')
    return ax


##
def showTextState(grid_world, grid_list, ax):
    for x in range(grid_world.shape[0]):
        for y in range(grid_world.shape[1]):
            if grid_world[x, y] >= -3000:
                ax.annotate(grid_list.get(grid_world[x, y]), xy=(y, x), horizontalalignment='center')


##
def showPolicy(policy, ax):
    for x in range(policy.shape[0]):
        for y in range(policy.shape[1]):
            if policy[x, y] == 0:
                ax.annotate('$\downarrow$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 1:
                ax.annotate(r'$\rightarrow$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 2:
                ax.annotate('$\uparrow$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 3:
                ax.annotate('$\leftarrow$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 4:
                ax.annotate('$\perp$', xy=(y, x), horizontalalignment='center')


##
def ValIter(R, discount, maxSteps, infHor, probModel=None):
    return 0

# YOUR CODE HERE


##
def maxAction(V, R, discount, probModel=None):
    return 0

# YOUR CODE HERE

##
def findPolicy(V, probModel=None):
    return 0

# YOUR CODE HERE

############################

saveFigures = True

data = genGridWorld()
grid_world = data[0]
grid_list = data[1]

# YOUR CODE HERE
probModel = ...

ax = showWorld(grid_world, 'Environment')
showTextState(grid_world, grid_list, ax)
if saveFigures:
    plt.savefig('gridworld.pdf')

# Finite Horizon
V = ValIter(...)
V = V[:, :, 0];
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon')
if saveFigures:
    plt.savefig('value_Fin_15.pdf')

policy = findPolicy(...)
ax = showWorld(grid_world, 'Policy - Finite Horizon')
showPolicy(policy, ax)
if saveFigures:
    plt.savefig('policy_Fin_15.pdf')

# Infinite Horizon
V = ValIter(...)
showWorld(np.maximum(V, 0), 'Value Function - Infinite Horizon')
if saveFigures:
    plt.savefig('value_Inf_08.pdf')

policy = findPolicy(...);
ax = showWorld(grid_world, 'Policy - Infinite Horizon')
showPolicy(policy, ax)
if saveFigures:
    plt.savefig('policy_Inf_08.pdf')

# Finite Horizon with Probabilistic Transition
V = ValIter(...)
V = V[:, :, 0];
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition')
if saveFigures:
    plt.savefig('value_Fin_15_prob.pdf')

policy = findPolicy(...)
ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition')
showPolicy(policy, ax)
if saveFigures:
    plt.savefig('policy_Fin_15_prob.pdf')
