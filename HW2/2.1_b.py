import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
T = 50
A_t = np.array([[1, 0.1], [0, 1]])
B_t = np.array([[0], [0.1]])
b_t = np.array([5, 0])
Sigma = np.array([[0.01, 0], [0, 0.01]])
K_t = np.array([5, 0.3])
k_t = 0.3
H_t = 1

epoch = 20

at_value = np.zeros((epoch, T+1))
st_value11 = np.zeros((epoch*2, T+1))
st_value22 = np.zeros((epoch*2, T+1))
rewards = np.zeros((epoch, T+1))

def SQR_X1():
    for i in range(0, epoch):
        # if t = 0,....,49
        # the first value of s_0 is normal distribution with mu=0, cov=identity
        s_t0 = np.random.multivariate_normal(np.array([0, 0]), np.identity(2))
        s_t0 = s_t0.reshape((2, 1))
        # update for s_t, a_t, and reward value
        for j in range(0, T+1):
            R_t = justify_Rt(j)
            r_t = justify_rt(j)
            w_t = np.random.multivariate_normal(b_t, Sigma)
            w_t = w_t.reshape((2, 1))
            size = [2 * i, 2 * i + 1]

            if j == 0:
                s_t = s_t0
            else:
                s_t = s_temp
            # for s_t^des is r_t
            a_t = K_t @ (r_t - s_t) + k_t
            at_value[i, j] = a_t
            rewards[i, j] = reward(s_t, r_t, R_t, a_t, H_t, j)
            st_value11[size, j:j+1] = s_t
            s_temp = A_t @ s_t + B_t * a_t + w_t

    return st_value11

def SQR_X2():
    for i in range(0, epoch):
        # if t = 0,....,49
        # the first value of s_0 is normal distribution with mu=0, cov=identity
        s_t0 = np.random.multivariate_normal(np.array([0, 0]), np.identity(2))
        s_t0 = s_t0.reshape((2, 1))
        # update for s_t, a_t, and reward value
        for j in range(0, T+1):
            R_t = justify_Rt(j)
            r_t = justify_rt(j)
            w_t = np.random.multivariate_normal(b_t, Sigma)
            w_t = w_t.reshape((2, 1))
            size = [2 * i, 2 * i + 1]

            if j == 0:
                s_t = s_t0
            else:
                s_t = s_temp
            # for s_t^des is 0
            a_t = K_t @ (0 - s_t) + k_t
            at_value[i, j] = a_t
            rewards[i, j] = reward(s_t, r_t, R_t, a_t, H_t, j)
            st_value22[size, j:j+1] = s_t
            s_temp = A_t @ s_t + B_t * a_t + w_t

    return st_value22





def justify_rt(t):
    if t <= 14:
        return np.array([[10], [0]])
    else:
        return np.array([[20], [0]])


def justify_Rt(t):
    if t == 14 or t == 40:
        return np.array([[1000000, 0], [0, 0.1]])
    else:
        return np.array([[0.01, 0], [0, 0.1]])


def reward(s, r, R, a, H, t):
    if t <= T - 1:
        # print('s', s.shape, 'r', r.shape, 'R', R.shape, 'a', a.shape)
        res = -(s - r).T @ R @ (s - r) - a * H * a  # shape(1, )
        return res
    else:
        res = -(s - r).T @ R @ (s - r)  # shape(1, )
        return res


st_value11 = SQR_X1()
st_value22 = SQR_X2()

"""
Pr(mu-2*std <= x >= mu+2*std) = 0.95
"""
# plot the x value of s_t, s_des = r_t
s_x1 = st_value11[::2, :]  # get s_t the x value of each epoch
mu_x1 = np.mean(s_x1, axis=0)
std_x1 = np.std(s_x1, axis=0)
plt.subplot(211)
x = np.linspace(0, T, T+1)
xx1 = mu_x1+2*std_x1/np.sqrt(epoch)
xx2 = mu_x1-2*std_x1/np.sqrt(epoch)
plt.plot(x, mu_x1, lw=0.5, label='walker position', color='blue')
plt.fill_between(x, xx1, xx2, facecolor='blue', alpha=0.5)
plt.title("controller of b): mean of sx, s_des is r_t")

# plot the y value of s_t, s_des = r_t
s_y1 = st_value11[1::2, :]  # get s_t the y value of each epoch
mu_y1 = np.mean(s_y1, axis=0)
std_y1 = np.std(s_y1, axis=0)
plt.subplot(211)
y = np.linspace(0, T, T+1)
yy1 = mu_y1+2*std_y1/np.sqrt(epoch)
yy2 = mu_y1-2*std_y1/np.sqrt(epoch)
plt.plot(y, mu_y1, lw=0.5, label='walker position', color='blue')
plt.fill_between(y, yy1, yy2, facecolor='blue', alpha=0.5)
plt.title("controller of b): mean of sy, s_des is r_t")


#  plot the x value of s_t, s_des = r_t
s_x2 = st_value22[::2, :]  # get s_t the y value of each epoch
mu_x2 = np.mean(s_x2, axis=0)
std_x2 = np.std(s_x2, axis=0)
plt.subplot(212)
x = np.linspace(0, T, T+1)
xx1 = mu_x2+2*std_x2/np.sqrt(epoch)
xx2 = mu_x2-2*std_x2/np.sqrt(epoch)
#print(mux.size, x.size)
plt.plot(x, mu_x2, lw=0.5, label='walker position', color='blue')
plt.fill_between(x, xx1, xx2, facecolor='blue', alpha=0.5)
plt.title("controller of b): mean of sx, s_des is 0")

s_y2 = st_value22[1::2, :]  # get s_t the y value of each epoch
mu_y2 = np.mean(s_y2, axis=0)
std_y2 = np.std(s_y2, axis=0)
plt.subplot(212)
y = np.linspace(0, T, T+1)
yy1 = mu_y2+2*std_y2/np.sqrt(epoch)
yy2 = mu_y2-2*std_y2/np.sqrt(epoch)
plt.plot(y, mu_y2, lw=0.5, label='walker position', color='blue')
plt.fill_between(y, yy1, yy2, facecolor='blue', alpha=0.5)
plt.title("controller of b): mean of sy,s_des is 0 ")

plt.show()

