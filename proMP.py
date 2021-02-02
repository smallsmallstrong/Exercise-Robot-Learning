import matplotlib.pyplot as plt
import numpy as np
from getImitationData import *
from getProMPBasis import *


def proMP(nBasis, condition=False):
    dt = 0.002
    time = np.arange(dt, 3, dt)
    nSteps = len(time)
    data = getImitationData(dt, time, multiple_demos=True)
    q = data[0]  # (45, 1499)
    qd = data[1]  # (45, 1499)
    qdd = data[2]  # (45, 1499)

    bandwidth = 0.2
    Phi = getProMPBasis(dt, nSteps, nBasis, bandwidth)  # (1499, 30)

    w = np.linalg.solve((Phi.T @ Phi), Phi.T) @ q.T

    mean_w = np.mean(w, axis=1)  # (30,)
    cov_w = np.cov(w)  # (30, 30)

    mean_traj = Phi @ mean_w  # (1499,)
    mean_traj.reshape(nSteps, 1)

    cov = Phi @ cov_w @ Phi.T
    std_traj = np.sqrt(cov.diagonal()).reshape(nSteps, )
    plt.figure()
    plt.hold('on')
    plt.fill_between(time, (mean_traj - 2 * std_traj), (mean_traj + 2 * std_traj), alpha=0.5, edgecolor='#1B2ACC',
                     facecolor='#089FFF')
    plt.plot(time, mean_traj, color='#1B2ACC')
    plt.plot(time, q.T)
    plt.title('ProMP with ' + str(nBasis) + ' basis functions')
    plt.savefig('h_N20.pdf')
    plt.draw_all()
    plt.pause(0.001)

    # Conditioning
    if condition:
        y_d = 3.0
        Sig_d = 0.0002
        t_point = int(np.round(2300 / 2))
        tmp = (cov_w @ Phi.T[:, t_point]) / (Sig_d + Phi.T[:, t_point] @ (cov_w @ Phi.T[:, t_point]))  # (30,1)
        tmp = tmp.reshape(30, 1)

        #print(Phi[1150, :].shape, '---', mean_w.shape)
        mean_w = mean_w.reshape(30, 1)
        mean_w_new = mean_w + tmp @ (y_d-((Phi[1150, :].reshape(1, 30))@(mean_w.reshape(30, 1))).reshape(1, 1))  # (30,1)

        cov_w_new = cov_w - tmp@((Phi[1150, :].reshape(1, 30))@ (cov_w.reshape(30, 30)))  # (30, 30)

        cov_t = Phi @ cov_w_new @ Phi.T

        mean_traj_new = Phi @ mean_w_new
        mean_traj_new = mean_traj_new.reshape(nSteps, )

        std_traj_new = np.sqrt(cov_t.diagonal())
        std_traj_new = std_traj_new.reshape(nSteps, )

        plt.figure()
        plt.hold('on')
        plt.fill_between(time, mean_traj - 2 * std_traj, mean_traj + 2 * std_traj, alpha=0.5, edgecolor='#1B2ACC',
                         facecolor='#089FFF',label='previous variance')
        plt.plot(time, mean_traj, color='#1B2ACC',label='previous mean')
        plt.legend('lowleft')
        plt.fill_between(time, mean_traj_new - 2 * std_traj_new, mean_traj_new + 2 * std_traj_new, alpha=0.5,
                         edgecolor='#CC4F1B', facecolor='#FF9848', label='new variance')
        plt.plot(time, mean_traj_new, color='#CC4F1B', label='new mean')
        plt.legend(loc=0)
        sample_traj = np.dot(Phi, np.random.multivariate_normal(mean_w_new.flatten(), cov_w_new, 10).T)
        plt.plot(time, sample_traj)
        plt.title('ProMP after contidioning with new sampled trajectories')
        #plt.savefig('i_new_trajectory.pdf')