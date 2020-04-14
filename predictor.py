import numpy as np
import matplotlib.pyplot as plt


class PredictorLinear:
    def __init__(self, mode, dt, t0):
        assert mode in ['last_obs', 'quad', 'linear_fit']
        self.mode = mode
        self.t = t0
        self.dt = dt

        # record
        self.t_traj = []
        self.pred_traj = []

    def predict(self, traj_past, t_pred, radius):
        if self.mode == 'last_obs':
            sx = float(traj_past[-1][0])
            sy = float(traj_past[-1][1])
            vx = float(traj_past[-1][2])
            vy = float(traj_past[-1][3])
            l_pred = int(t_pred / self.dt)
            traj_pred = np.empty((l_pred+1, 5))
            for i in range(l_pred+1):
                traj_pred[i, :] = np.array([sx + i * vx * self.dt, sy + i * vy * self.dt, vx, vy, radius])
        elif self.mode == 'quad':
            # todo @ need implementation
            raise Exception('quad mode is not available yet.')
        elif self.mode == 'linear_fit':
            # todo @ need implementation
            raise Exception('linear mode is not available yet.')
        else:
            raise Exception('Invalid mode fo Linear Predictor')

        self.t_traj.append(self.t)
        self.pred_traj.append(traj_pred)
        self.t = self.t + self.dt

        return traj_pred


# TEST
# mode 'last_obs' passed
# mode 'quad' todo
# mode 'linear_fit' todo

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    dt = 0.1
    var = 0.8 # percent
    sxs = np.arange(5) * dt + np.random.random(5) * var * dt
    sys = np.arange(5) * 0.1 * dt + np.random.random(5) * 0.1 * var * dt
    vxs = (sxs - np.roll(sxs, 1)) / dt
    vys = (sys - np.roll(sys, 1)) / dt
    vxs[0] = 0.0
    vys[0] = 0.0

    traj_history = np.array([sxs, sys, vxs, vys]).swapaxes(0, 1)

    mode = 'last_obs'
    predictor = PredictorLinear(mode=mode, dt=dt, t0=0)
    pred = predictor.predict(traj_past=traj_history, t_pred=1, radius=2.7)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(traj_history[:, 0], traj_history[:, 1], '.-k', label='observed')
    ax.plot(pred[:, 0], pred[:, 1], '.-r', label='predicted')
    ax.axis('equal')
    ax.legend()
    ax.set_title(f'Test of mode \'{mode}\', dt={dt:.2f}s')
    plt.show()