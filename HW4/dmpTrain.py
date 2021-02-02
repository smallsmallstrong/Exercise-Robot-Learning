# Learns the weights for the basis functions.
#
# Q_IM, QD_IM, QDD_IM are vectors containing positions, velocities and
# accelerations of the two joints obtained from the trajectory that we want
# to imitate.
#
# DT is the time step.
#
# NSTEPS are the total number of steps.

"""c) Double Pendulum - Training"""

from getDMPBasis import *
import matplotlib.pyplot as plt

class dmpParams():
    def __init__(self):
        self.alphaz = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.Ts = 0.0
        self.tau = 0.0
        self.nBasis = 0.0
        self.goal = 0.0
        self.w = 0.0

# (2,1499),(2,1499),(2,1499), 0.002,
# 1499
def dmpTrain(q, qd, qdd, dt, nSteps):
    # print(q,'----',qd.shape,'----',qdd.shape)
    params = dmpParams()
    # Set dynamic system parameters
    params.alphaz = (nSteps * dt-dt) / 3.0
    params.alpha = 25.0
    params.beta = 6.25
    params.Ts = nSteps * dt-dt
    params.tau = 1.0
    params.nBasis = 50
    params.goal = q[:, -1]
    # (1499,50)
    Phi = getDMPBasis(params, dt, nSteps)

    # Compute the forcing function (2,1499)
    f = np.zeros((2, nSteps))
    for i in range(nSteps-1):

        temp = (params.goal).reshape(2, 1)-(q[:, [i]]).reshape(2, 1)
        ft = qdd[:, [i]]/(params.tau**2) - params.alpha*(params.beta*temp-qd[:, [i]]/params.tau)
        f[:, [i]] = ft.reshape(2, 1)

    params.w = np.linalg.solve((Phi.T@Phi), Phi.T) @ f.T


    return params


