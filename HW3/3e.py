"""
This is for 3e), Comparison between training set and validation set
Date: 15.12.2020
Author: Yulian Sun
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la


# read the txt file
file_train = np.loadtxt('data_ml/training_data.txt', dtype=float)
file_vali = np.loadtxt('data_ml/validation_data.txt', dtype=float)
file_train = np.array(file_train); file_vali = np.array(file_vali)
input_train = file_train[0, :]; input_vali = file_vali[0, :]
output_train = file_train[1, :]; output_vali = file_vali[1, :]
_, size = file_train.shape


# calculate features phi(x)
def phi_x(n, x):
    size = x.size
    phi = np.zeros((n, size))
    for i in range(n):
        for j in range(size):
            # if x[i] == 0.0: x[i] = 1e26
            phi[i][j] = np.sin(pow(2, i) * x[j])
    return phi


# calculate parameters theta
def calu_theta(n, x, y, s):
    phi = phi_x(n, x)
    theta = la.solve(((phi @ phi.T)), phi) @ y
    return theta


# rmse
def RMSE(a, b):# calculate RMSE
    return np.sqrt(((a - b) ** 2).mean())


def train():
    thetalist = []
    rmselist_train = []
    rmselist_vali = []
    temp_y_train = np.zeros((9, size))
    temp_y_vali = np.zeros((9, size))
    for i in range(9):
        thetalist.append(calu_theta(i+1, input_train, output_train, size))

        temp_y_train[i, :] = thetalist[i].T @ phi_x(i+1, input_train)
        temp_y_vali[i, :] = thetalist[i].T @ phi_x(i+1, input_vali)
            #print(temp_y_train[i, j])

    for s in range(9):
        rmselist_train.append(RMSE(temp_y_train[s, :], output_train))
        rmselist_vali.append(RMSE(temp_y_vali[s, :], output_vali))
    return rmselist_train, rmselist_vali


# best_theta_2, best_theta_3, best_theta_9 = train()
# print(best_theta_2, best_theta_3, best_theta_9)


rmse_error1, rmse_error2 = train()
x = np.linspace(1, 9, 9)
print(rmse_error2,'----',x)
# plt.subplot(211)
plt.plot(x, rmse_error1, label='RMSE in training set')
plt.legend(loc='upper right')
# plt.subplot(212)
plt.plot(x, rmse_error2, label='RMSE in validation set')
plt.legend(loc='upper right')
plt.legend()
plt.show()
print(np.min(rmse_error2))
