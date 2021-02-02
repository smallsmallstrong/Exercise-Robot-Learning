import warnings

from dmpComparison import *
from proMP import *

warnings.filterwarnings("ignore")

plt.ion()

# Reproduce the desired trajectory with a DMP and save a plot
#dmpComparison([], [], 'dmp')

# Reproduce the trajectory and condition on the goal position
#dmpComparison([[0, 0.2], [0.8, 0.5]], [], 'goalCond')
# #
# # # Reproduce the trajectory and condition on the time
#dmpComparison([], [0.5, 1.5], 'timeCond')
#
# # Learn a ProMP with 10 radial basis functions
proMP(20)
# condition = True
# proMP(30, condition)
input('Finish?')

