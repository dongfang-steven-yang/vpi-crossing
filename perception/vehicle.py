import numpy as np


class PredictorLinear:
    def __init__(self, dt, t0):
        self.t = t0
        self.dt = dt

        # record
        self.t_traj = []
        self.pred_traj = []

    def predict(self, traj_past, t_pred, radius):

        sx = float(traj_past[-1][0])
        sy = float(traj_past[-1][1])
        vx = float(traj_past[-1][2])
        vy = float(traj_past[-1][3])
        l_pred = int(t_pred / self.dt)
        traj_pred = np.empty((l_pred+1, 5))
        for i in range(l_pred+1):
            traj_pred[i, :] = np.array([sx + i * vx * self.dt, sy + i * vy * self.dt, vx, vy, radius])

        self.t_traj.append(self.t)
        self.pred_traj.append(traj_pred)
        self.t = self.t + self.dt

        return traj_pred