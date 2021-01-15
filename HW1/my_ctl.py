# CTL is the name of the controller.
# Q_HISTORY is a matrix containing all the past position of the robot. Each row of this matrix is [q_1, ... q_i], where
# i is the number of the joints.
# Q and QD are the current position and velocity, respectively.
# Q_DES, QD_DES, QDD_DES are the desired position, velocity and acceleration, respectively.
# GRAVITY is the gravity vector g(q).
# CORIOLIS is the Coriolis force vector c(q, qd).
# M is the mass matrix M(q).

import numpy as np


def my_ctl(ctl, q, qd, q_des, qd_des, qdd_des, q_hist, q_deshist, gravity, coriolis, M):
    KP = np.diag([60, 30])*10
    KD = np.diag([10, 6])*10
    KI = np.diag([0.1, 0.1])*10
    t_end = 3.0
    dt = 0.002
    # coriolis = np.array(coriolis).reshape((2, 1))
    # print(q_hist.shape, '--', q_deshist.shape)
    if ctl == 'P':
        u = np.zeros((2, 1))  # Implement your controller here
        u = KP @ (q_des - q)
        u = u.reshape((2, 1))
        print(u)
    elif ctl == 'PD':
        u = np.zeros((2, 1))  # Implement your controller here
        u = KP @ (q_des - q) + KD @ (qd_des - qd)
        u = u.reshape((2, 1))
    elif ctl == 'PID':
        u = np.zeros((2, 1))  # Implement your controller here
        # u = np.array([60 * (q_des[0] - q[0]) + 10 * (qd_des[0] - qd[0]) +
        #               0.1 * (sum(q_deshist[:, 0] - q_hist[:, 0]) + (q_des[0] - q[0])) * 0.002,
        #               30 * (q_des[1] - q[1]) + 6 * (qd_des[1] - qd[1]) +
        #               0.1 * (sum(q_deshist[:, 1] - q_hist[:, 1]) + (q_des[1] - q[1])) * 0.002]).reshape(-1, 1)
        u = KP @ (q_des - q) + KD @ (qd_des - qd) + KI @ np.array([0, t_end]) * dt
        u = u.reshape((2, 1))
    elif ctl == 'PD_Grav':
        u = np.zeros((2, 1))  # Implement your controller here
        u = KP @ (q_des - q) + KD @ (qd_des - qd) + gravity
        u = u.reshape((2, 1))
    elif ctl == 'ModelBased':
        u = np.zeros((2, 1))  # Implement your controller here
        qddref = qdd_des + KD @ (qd_des-qd) + KP @ (q_des-q)
        u = M @ qddref + coriolis + gravity
        u = u.reshape((2, 1))
    return u
