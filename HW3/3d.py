"""
This is for 3d), root mean square error
Date: 15.12.2020
Author: Yulian Sun
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la


# read the txt file
file = np.loadtxt('data_ml/training_data.txt', dtype=float)
file = np.array(file)
input = file[0, :]
output = file[1, :]
_, size = file.shape


# calculate features phi(x)
def phi_x(n, x):
    size = x.size
    phi = np.zeros((n, size))
    for i in range(n):
        for j in range(size):
            #if x[i] == 0.0: x[i] = 1e26
            phi[i][j] = np.sin(pow(2, i)*x[j])
    return phi


# calculate parameters theta
def calu_theta(n, x, y):
    phi = phi_x(n, x)
    theta = la.solve(((phi @ phi.T)), phi) @ y
    return theta


# lse
def least_square_error(a, b):
    return np.sum(pow((a-b), 2))


# rmse
def RMSE(a, b):# calculate RMSE
    return np.sqrt(((a - b) ** 2).mean())


def train():
    thetalist = []
    rmselist = []
    temp_y = np.zeros((9, size))
    min_ls = 1000
    index = 0
    for i in range(9):
        thetalist.append(calu_theta(i+1, input, output))
        #for j in range(size):
        temp_y[i, :] = thetalist[i].T @ phi_x(i+1, input)
        #print(temp_y[i, :])
    for s in range(9):
        error = RMSE(temp_y[s, :], output)
        rmselist.append(error)
    print(min(rmselist))
    return rmselist


# best_theta_2, best_theta_3, best_theta_9 = train()
# print(best_theta_2, best_theta_3, best_theta_9)


rmse_error = train()
x = np.linspace(1, 9, 9)
print(rmse_error,'----',x)
plt.plot(x, rmse_error, label='RMSE for different models')

plt.legend()
plt.show()
