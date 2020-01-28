import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()


class PredictorReachableSet:
    def __init__(self, dt, acc_limit, t_pred, ped_R):
        self.dt = dt
        self.acc_limit = acc_limit
        self.t_pred = t_pred
        self.ped_R = ped_R

    def predict(self, mode, traj_pos_backward, traj_class_backward):

        # current position
        sx = float(traj_pos_backward[0][0])
        sy = float(traj_pos_backward[0][1])

        # current speed
        if len(traj_pos_backward) == 1:
            # change of classification:
            if traj_class_backward[0] == 1 or traj_class_backward[0] == 2:
                # vehicle/cyclist: assign a goal in front, here assumed a lower speed vec pointing to front
                vx = 2.0
                vy = 0.0
            else:
                # unknown static obstacle or pedestrian
                vx = 0.0
                vy = 0.0
        else:
            # speed is estimated based on previous location
            # todo @ can improve in the future work
            if mode is 'uber_mode':
                if traj_class_backward[0] == 1 or traj_class_backward[0] == 2:
                    # vehicle/cyclist: assign a goal in front, here assumed a lower speed vec pointing to front
                    vx = 2.0
                    vy = 0.0
                else:
                    # unknown static obstacle or pedestrian
                    vx = 0.0
                    vy = 0.0
            else:
                vx = float((traj_pos_backward[0][0] - traj_pos_backward[1][0]) / self.dt)
                vy = float((traj_pos_backward[0][1] - traj_pos_backward[1][1]) / self.dt)

        # predict
        if mode is 'linear_vel' or mode is 'uber_mode':
            acc_limit = 0.0
        elif mode is 'reach_acc':
            acc_limit = self.acc_limit
        else:
            raise Exception('Invalid predictor mode !')
        return np.array(eng.point_mass_reachable(sx, sy, vx, vy, acc_limit, self.t_pred, self.ped_R, self.dt))