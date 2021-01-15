"""
This is for 3h), Kernel Regression
Date: 16.12.2020
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

# get x values for prediction
x_values = np.linspace(0.0, 6, num=601)
delta = 0.15


# rmse
def RMSE(a, b): # calculate RMSE
    return np.sqrt(((a - b) ** 2).mean())


# exponential squared kernel
def exponential_square_kernel(xi, xj):
    kernel_K = np.exp(-((xi-xj)**2)/delta**2)
    return kernel_K


# calculate kernel K_ij
def kernel_K(X):
    N = X.size
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = exponential_square_kernel(X[i], X[j])
    return K


# calulate alpha(like phi in linear regression)
def calu_alpha(X,y):
    N = X.size
    K = kernel_K(X)
    alpha = la.solve(K,y)
    return alpha


# calculate kernel k_i
def kernel_k(x, X):
    N = X.size
    k = np.zeros((N, 1))
    for i in range(N):
        k[i] = exponential_square_kernel(x, X[i])
    return k


# calculate high dimensional k kernel
def calu_kernel_k(xx,X):
    m = xx.size
    N = X.size
    kk = np.zeros((N, m))
    for i in range(m):
        kk[:, [i]] = kernel_k(xx[i], X)
    return kk


def test():
    m = x_values.size
    #y_hat = np.zeros((m, 1))
    kk = calu_kernel_k(x_values, input_train)
    alpha = calu_alpha(input_train,output_train)
    y_hat = kk.T @ alpha
    return y_hat


def validation():
    m = input_vali.size
    kk = calu_kernel_k(input_vali, input_train)
    alpha = calu_alpha(input_train, output_train)
    y_hat = kk.T @ alpha
    error = RMSE(y_hat,output_vali)
    return error

y_hat = test()
error = validation()
print(y_hat, '---\n', error)
# plt.subplot(211)
plt.plot(x_values, y_hat, label='kernel regression for prediction')
plt.plot(input_train,output_train, '+', color='black',label='original data in training set')
plt.legend(loc='upper right')

plt.legend()
plt.show()
