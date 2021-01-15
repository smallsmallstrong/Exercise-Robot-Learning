"""
This is for 3f), K-fold cross validation
Date: 16.12.2020
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
K = size


# preprocess file as k-1 training set
def preprocess_k_fold():
    fold_input = np.zeros((K, K-1)) # shape is 30x29, one line --> one k-1 fold
    fold_output = np.zeros((K, K-1))
    for i in range(K):
        copy_input = np.copy(input)
        copy_output = np.copy(output)
        fold_input[i, :] = np.delete(copy_input, [i], None)
        fold_output[i, :] = np.delete(copy_output, [i], None)
    return fold_input, fold_output


# calculate features phi(x)
def phi_x(n, x):
    size = x.size
    phi = np.zeros((n, size))
    for i in range(n):
        for j in range(size):
            #if x[i] == 0.0: x[i] = 1e26
            phi[i][j] = np.sin(pow(2, i)*x[j])
    return phi


# calculate features phi(x) for one point
def phi_x_for_one(n, xi):
    phi = np.zeros((n, 1))
    for i in range(n):
        phi[i, :] = np.sin((2**i)*xi)
    return phi

# calculate parameters theta
def calu_theta(n, x, y):
    phi = phi_x(n, x)
    theta = la.solve(((phi @ phi.T)), phi) @ y
    return theta


# rmse
def RMSE(a, b):# calculate RMSE
    return np.sqrt(((a - b) ** 2).mean())
fold_in, fold_out = preprocess_k_fold()


# leave-one-out cross validation
def loo(n):
    errlist = np.zeros((K, 1))
    temp_y_train = np.zeros((K, 1))
    thetalist = []
    for i in range(K):
        thetalist.append(calu_theta(n, fold_in[i, :], fold_out[i, :]))
        temp_y_train[i, :] = thetalist[i].T @ phi_x_for_one(n, input[i])
        errlist[i] = RMSE(temp_y_train[i], output[i])
    return errlist


def train():
    #loolist = np.zeros((K, 9))
    meanlist = np.zeros((1, 9))
    varlist = np.zeros((1, 9))
    stdlist = np.zeros((1, 9))
    for i in range(9):
        #loolist[:, [i]] = loo(i+1)
        meanlist[:, [i]] = np.mean(loo(i+1))
        varlist[:, [i]] = np.var(loo(i+1))
        stdlist[:, [i]] = np.std(loo(i+1))
    return meanlist, varlist, stdlist


mean, var, std= train()
# print(np.min(mean),'----',np.min(var))
# print(mean,'\n\n',var)
x = np.linspace(1, 9, 9)
yy1 = mean+1.96*std/np.sqrt(size)
yy2 = mean-1.96*std/np.sqrt(size)

plt.plot(x, mean.flatten(), lw=0.5, label='mean for different models', color='black')
# dy = 2*np.sqrt(var)/(np.sqrt(30))
# print(dy)
# plt.errorbar(x, mean, yerr=dy, color='b')
plt.fill_between(x, yy1.flatten(), yy2.flatten(), label='95% CI for different models', facecolor='blue', alpha=0.5)

# plt.subplot(211)
# plt.hist(mean.flatten(), bins=9,  label='mean of the RMSE')
# plt.legend(loc='upper right')
# plt.subplot(212)
# plt.hist(var.flatten(), bins=9, label='variance of the RMSE')
plt.legend(loc='upper right')
plt.show()
