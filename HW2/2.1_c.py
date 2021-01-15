import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
T = 50
A_t = np.array([[1, 0.1], [0, 1]])
B_t = np.array([[0], [0.1]])
b_t = np.array([5, 0])
Sigma = np.array([[0.01, 0], [0, 0.01]])
Kt = np.array([5, 0.3])
kt = 0.3
H_t = 1

epoch = 20

st_value = np.zeros((epoch*2, T+1))
rewards = np.zeros((epoch, T+1))
st_value11 = np.zeros((epoch*2, T+1))
st_value22 = np.zeros((epoch*2, T+1))
Kt_value = np.zeros((T+1, epoch*2))
kt_value = np.zeros((epoch, T+1))
# Vt_value = np.zeros((epoch*2, T+1))
# vt_value = np.zeros((epoch, T+1))
def SQR():
    for i in range(0, epoch):
        # if t = 0,....,50
        # the first value of s_des is 0
        s_t0 = np.random.multivariate_normal(np.array([0, 0]), np.identity(2))
        s_t0 = s_t0.reshape((2, 1))
        # update for s_t, a_t, and reward value
        for j in range(T, -1, -1):
            #t = T-j
            r_t = justify_rt(j)
            R_t = justify_Rt(j)
            w_t = np.random.multivariate_normal(b_t, Sigma)
            w_t = w_t.reshape((2, 1))
            size = [2 * i, 2 * i + 1]
            if j == T:
                #s_t = r_t
                V_t = R_t
                v_t = R_t @ r_t
                Kt_value[j, size] = np.zeros((1, 2))
                kt_value[i, j] = 0
                M_t = B_t * (H_t + B_t.T @ V_t @ B_t) ** (-1) * B_t.T @ V_t @ A_t
            else:
                K_t = -(H_t + B_t.T @ V_t @ B_t) ** (-1) * B_t.T @ V_t @ A_t
                bt2 = b_t.reshape((2, 1))
                k_t = -(H_t + B_t.T @ V_t @ B_t) ** (-1) * B_t.T @ (V_t @ bt2 - v_t)
                Kt_value[j, size] = K_t
                kt_value[i, j] = k_t
                v_t = R_t @ r_t + (A_t - M_t).T @ (v_t - V_t @ bt2)
                V_t = R_t + (A_t - M_t).T @ V_t @ A_t
                M_t = B_t * (H_t + B_t.T @ V_t @ B_t) ** (-1) * B_t.T @ V_t @ A_t


        for s in range(0, T+1):
            r_t = justify_rt(s)
            R_t = justify_Rt(s)
            s_t0 = np.random.multivariate_normal(np.array([0, 0]), np.identity(2))
            s_t0 = s_t0.reshape((2, 1))
            size = [2 * i, 2 * i + 1]
            if s == 0:
                s_t = s_t0
            else:
                s_t = s_temp
            Kt = Kt_value[s, size]
            kt = kt_value[i, s]
            a_t = Kt @ s_t + kt
            w_t = np.diag(np.random.normal(b_t, Sigma)).reshape(2, 1)
            rewards[i, s] = reward(s_t, r_t, R_t, a_t, H_t, s)
            st_value[size, s: s+1] = s_t
            s_temp = A_t @ s_t + B_t * a_t + w_t

    return st_value, rewards

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
            a_t = Kt @ (r_t - s_t) + kt
            #at_value[i, j] = a_t
            #rewards[i, j] = reward(s_t, r_t, R_t, a_t, H_t, j)
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
            a_t = Kt @ (0 - s_t) + kt
            #at_value[i, j] = a_t
            #rewards[i, j] = reward(s_t, r_t, R_t, a_t, H_t, j)
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


st_value, rewards = SQR()
st_value11 = SQR_X1()
st_value22 = SQR_X2()
reward_cum = np.cumsum(rewards, axis=1)
# print(reward_cum)
mu_r = np.mean(reward_cum)
std_r = np.std(reward_cum)
print('The mean of cumulative reward is :', mu_r, ',The standard derivation of accumulation reward is :', std_r)
# print('The mean is:', mulist, ', The standard derivation is:', stdlist)
# print(res_mu)
"""
Pr(mu-2*std <= x >= mu+2*std) = 0.95
"""
# plot the x value of s_t
s_x = st_value[::2, :]  # get s_t the x value of each epoch
mu_x = np.mean(s_x, axis=0)
std_x = np.std(s_x, axis=0)
plt.subplot(211)
x = np.linspace(0, T, T+1)
xx1 = mu_x+2*std_x/np.sqrt(epoch)
xx2 = mu_x-2*std_x/np.sqrt(epoch)
plt.plot(x, mu_x, lw=0.5, label='walker position', color='blue')
plt.fill_between(x, xx1, xx2, facecolor='blue', alpha=0.5)
plt.title("controller of c): mean of sx")

s_x = st_value11[::2, :]  # get s_t the x value of each epoch
mu_x = np.mean(s_x, axis=0)
std_x = np.std(s_x, axis=0)
plt.subplot(211)
x = np.linspace(0, T, T+1)
xx1 = mu_x+2*std_x/np.sqrt(epoch)
xx2 = mu_x-2*std_x/np.sqrt(epoch)
plt.plot(x, mu_x, lw=0.5, label='walker position', color='orange')
plt.fill_between(x, xx1, xx2, facecolor='orange', alpha=0.5)
plt.title("controller of c): mean of sx")

s_x = st_value22[::2, :]  # get s_t the x value of each epoch
mu_x = np.mean(s_x, axis=0)
std_x = np.std(s_x, axis=0)
plt.subplot(211)
x = np.linspace(0, T, T+1)
xx1 = mu_x+2*std_x/np.sqrt(epoch)
xx2 = mu_x-2*std_x/np.sqrt(epoch)
plt.plot(x, mu_x, lw=0.5, label='walker position', color='yellow')
plt.fill_between(x, xx1, xx2, facecolor='yellow', alpha=0.5)
plt.title("controller of c): mean of sx")

# plot the y value of s_t
s_y = st_value[1::2, :]  # get s_t the y value of each epoch
mu_y = np.mean(s_y, axis=0)
std_y = np.std(s_y, axis=0)
plt.subplot(212)
y = np.linspace(0, T, T+1)
yy1 = mu_y+2*std_y/np.sqrt(epoch)
yy2 = mu_y-2*std_y/np.sqrt(epoch)
plt.plot(y, mu_y, lw=0.5, label='walker position', color='blue')
plt.fill_between(y, yy1, yy2, facecolor='blue', alpha=0.5)
plt.title("controller of c): mean of sy")

s_y = st_value11[1::2, :]  # get s_t the y value of each epoch
mu_y = np.mean(s_y, axis=0)
std_y = np.std(s_y, axis=0)
plt.subplot(212)
y = np.linspace(0, T, T+1)
yy1 = mu_y+2*std_y/np.sqrt(epoch)
yy2 = mu_y-2*std_y/np.sqrt(epoch)
plt.plot(y, mu_y, lw=0.5, label='walker position', color='orange')
plt.fill_between(y, yy1, yy2, facecolor='orange', alpha=0.5)
plt.title("controller of c): mean of sy")

s_y = st_value22[1::2, :]  # get s_t the y value of each epoch
mu_y = np.mean(s_y, axis=0)
std_y = np.std(s_y, axis=0)
plt.subplot(212)
y = np.linspace(0, T, T+1)
yy1 = mu_y+2*std_y/np.sqrt(epoch)
yy2 = mu_y-2*std_y/np.sqrt(epoch)
plt.plot(y, mu_y, lw=0.5, label='walker position', color='yellow')
plt.fill_between(y, yy1, yy2, facecolor='yellow', alpha=0.5)
plt.title("controller of c): mean of sy")

# plot the reward
# plt.subplots(1)
# r = np.linspace(0, epoch, epoch)
# plt.plot(r, reward_cum[:, -1], lw=0.5, label='walker position', color='blue')
# plt.title("result of reward")

plt.show()

