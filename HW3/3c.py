"""
This is for 3c), Least square error
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
    theta = la.solve(((phi@phi.T)), phi) @ y
    return theta


# lse
def least_square_error(a, b):
    return np.sum(pow((a-b), 2))


# get x values for prediction
x_values = np.linspace(0.0, 6, num=601)


def train():
    # n=2
    theta_2 = calu_theta(2, input, output)
    # n=3
    theta_3 = calu_theta(3, input, output)
    # n=9
    theta_9 = calu_theta(9, input, output)

    #print(phi_x(2, input).shape)
    temp2_y = theta_2.reshape(1, 2) @ phi_x(2, input)

    temp3_y = theta_3.reshape(1, 3) @ phi_x(3, input)

    temp9_y = theta_9.reshape(1, 9) @ phi_x(9, input)


    return theta_2, theta_3, theta_9


# best_theta_2, best_theta_3, best_theta_9 = train()
# print(best_theta_2, best_theta_3, best_theta_9)


def test():
    # y_2 = np.zeros((601, 1))
    # y_3 = np.zeros((601, 1))
    # y_4 = np.zeros((601, 1))
    # y_9 = np.zeros((601, 1))
    theta_2, theta_3,theta_9 = train()
    print(theta_2)
    y_2 = theta_2.T @ phi_x(2, x_values)
    print(y_2)
    print(phi_x(2, x_values))
    y_3 = theta_3.T @ phi_x(3, x_values)
    y_9 = theta_9.T @ phi_x(9, x_values)
    return y_2, y_3, y_9


y_2, y_3, y_9 = test()

# theta_2 = calu_theta(2, input, output)
# print(theta_2)
# plt.subplot(211)
# plt.plot(input, output, label='original data')
# plt.legend()
# # plt.show()
# plt.subplot(212)
plt.plot(x_values, y_2, label='2 features')
plt.plot(x_values, y_3, label='3 features')
plt.plot(input, output, '+', color='black',label='original data')

plt.plot(x_values, y_9, label='9 features')
plt.legend()
plt.show()
