import numpy as np
import matplotlib.pyplot as plt

def getProMPBasis(dt, nSteps, n_of_basis, bandwidth):
    # (2, 1499)
    time = np.arange(dt, nSteps * dt, dt)
    nBasis = n_of_basis
    T = nSteps * dt

    C = np.random.uniform(-2*bandwidth, T+2*bandwidth, nBasis)

    #X = 1  # Canonical system
    Phi = np.zeros((nSteps, nBasis))

    for k in range(nSteps):
        for j in range(nBasis):
            Phi[k, j] = np.exp(-0.5 * (time[k] - C[j]) ** 2/bandwidth ** 2)  # Basis function activation over time
        Phi[k, :] = (Phi[k, :] * time[k]) / np.sum(Phi[k, :])  # Normalize basis functions and weight by canonical state

    return Phi

#show the basis functions
# dt = 0.002
# nSteps = 1499
# N = 30
# bandwidth = 0.2
# time = np.arange(dt, nSteps * dt, dt)
# Phi = getProMPBasis(dt, nSteps, N, bandwidth)
# print(Phi)
# plt.plot(time, Phi)
# plt.savefig('f_basis.pdf')
# #plt.legend()
# plt.show()
