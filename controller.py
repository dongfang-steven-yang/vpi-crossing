import numpy as np
import cvxpy as cp

params_control_general = {
    # constraints
    'v_max': 22.5,
    'v_min': 0.0,
    'u_max': 7.0,
    'u_min': -7.0,
    'du_max': 5.0,
    'du_min': -5.0,
    'd_safe': 3.0,
}


# Model Predictive Controller -----------------------------------------------------------------------------------------
params_control_mpc = {
    # prediction step
    'N_pred': 15,
    # weights for cost
    'w_state': 1.0,
    'w_control': 1.0
}


class ModelPredictiveController:
    def __init__(self, vehicle, params, verbose=False):
        self.vehicle = vehicle
        self.dt = vehicle.dt
        self.t = vehicle.t
        self.verbose = verbose

        # parameters
        self.n_state = vehicle.A.shape[1]
        self.n_control = vehicle.B.shape[1]
        self.N = params['N_pred']  # prediction horizon
        self.d_safe = params['d_safe'] #
        self.W_lane = params['W_lane']

        # constraints
        self.v_max = params['v_max']
        self.v_min = params['v_min']
        self.u_max = params['u_max']
        self.u_min = params['u_min']
        self.du_max = params['du_max']
        self.du_min = params['du_min']

        # cost weights
        self.w_state = params['w_state']
        self.w_control = params['w_control']

        # record
        self.t_traj = []
        self.u_traj = []
        self.feasible_traj = []
        self.target_traj = []

    def generate_control(self, ref_speed, obj_pred):
        # check if pred length matches
        assert len(obj_pred) == self.N + 1

        # initial condition
        x_0 = self.vehicle.state.reshape(2)
        u_last = self.vehicle.u_last

        # data
        targets = self._get_closest_dist(pred=obj_pred, pos_veh_front=x_0[0] + self.vehicle.C2F)
        if self.verbose:
            print('--> distances to the closest obstacles for all prediction steps:')
            print(targets - x_0[0])

        # variables
        x = cp.Variable((self.n_state, self.N+1))
        u = cp.Variable((self.n_control, self.N))

        # cost and constraints
        cost = 0.0
        constr = []

        for t in range(self.N):
            cost += self.w_state * cp.sum_squares(x[1, t + 1] - ref_speed) + self.w_control * cp.sum_squares(u[:, t])
            constr += [
                x[:, t + 1] == self.vehicle.A@x[:, t] + self.vehicle.B@u[:, t],  # dynamics
                x[1, t + 1] <= self.v_max,
                x[1, t + 1] >= self.v_min, # vel constraint
                u[:, t] <= self.u_max,
                u[:, t] >= self.u_min,  # acc constraint
            ]
            # safe distance
            if targets[t + 1] < 1000:
                constr += [targets[t + 1] - x[0, t + 1] >= self.vehicle.C2F + self.d_safe]
            # acc rate constraint
            if t > 0:
                constr += [u[:, t] - u[:, t - 1] >= self.du_min * self.dt,
                           u[:, t] - u[:, t - 1] <= self.du_max * self.dt]
            else:
                constr += [u[:, t] - u_last >= self.du_min * self.dt,
                           u[:, t] - u_last <= self.du_max * self.dt]

        # initial condition
        constr += [x[:, 0] == x_0]

        # terminal condition
        if targets[self.N] < 1000:
            constr += [targets[self.N] - (x[0, self.N] + self.vehicle.C2F) >=
                       (x[1, self.N] / 2) * self.v_max / abs(self.u_min) + self.d_safe]
        #     constr += [x[1], self.N] == 0

        # construct problem
        problem = cp.Problem(cp.Minimize(cost), constr)
        # problem.solve(solver=cp.SCS, warm_start=True, verbose=True)
        problem.solve(solver=cp.ECOS, warm_start=True, verbose=False)
        # debug
        if self.verbose:
            print('--> Predicted vector of states X =')
            print(x.value)
            print('--> Predicted vector of control actions U =')
            print(u.value)
        # self.vehicle.A.dot(x_0.reshape(2, 1)) + self.vehicle.B.dot(np.array(1))

        # first control
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            action = u[0, 0].value
            print(f'-> MPC found solution u = {action}')
            u_out = action
            feasible = True

        else:
            print(f'-> MPC infeasible, apply emergency braking (maximum deceleration).')

            if self.verbose:
                print('------------------------------------')
                print('Test emergency braking to verify MPC solution:')
                xx = x_0.reshape(2, 1)
                uu = u_last
                for i in range(20):
                    uu = max(uu - 0.5, -7.0)
                    print(f'u = {uu}')
                    xx = self.vehicle.A.dot(xx) + self.vehicle.B.dot(np.array(uu).reshape(1, 1))
                    print(f'x = {xx.reshape(2)}')
                    print(f'at step {i+1} dist to predicted obs = {targets[i+1]:.4f} - {xx[0][0]:.4f} = {targets[i+1] - xx[0][0]:.4f}')

            u_out = max(self.u_min, u_last + self.du_min * self.dt)
            feasible = False

        # record
        self.t_traj.append(self.vehicle.t)
        self.u_traj.append(u_out)
        self.feasible_traj.append(feasible)
        self.target_traj.append(targets)
        self.t = self.t + self.dt

        return u_out, feasible

    def _get_closest_dist(self, pred, pos_veh_front):
        closest_dist = np.full(len(pred), float('Inf'))
        if pos_veh_front < pred[0][0]: # before the vehicle reach the crossing point
            for i in range(len(pred)):
                lat_max = pred[i, 1] + pred[i, 2]
                lat_min = pred[i, 1] - pred[i, 2]

                if 0 < lat_min < self.W_lane or 0 < lat_max < self.W_lane or \
                        (0 < lat_min and lat_max < self.W_lane) or (lat_min < 0 and self.W_lane < lat_max):
                    lon_min = pred[i, 0] - pred[i, 2]
                    closest_dist[i] = lon_min

        return closest_dist


# PID Controller -------------------------------------------------------------------------------------------------------

params_control_pid = {
    'K_P_speed': 1.0,
    'K_I_speed': 0.1,
    'K_D_speed': 0.0,
    'K_P_dist': 1.0,
}


class PIDController:
    def __init__(self, vehicle, params, verbose=False):
        # linked objects
        self.vehicle = vehicle

        # parameters
        self.dt = vehicle.dt
        self.t = vehicle.t
        self.verbose = verbose
        self.K_P_dist = params['K_P_dist']
        self.K_P_speed = params['K_P_speed']
        self.K_I_speed = params['K_I_speed']
        self.K_D_speed = params['K_D_speed']
        self.W_lane = params['W_lane']
        if self.K_D_speed == 0 and self.verbose:
            print('The differential term is not implemented. Only PI is applied.')
        self.d_safe = params['d_safe']

        # constraints
        self.v_max = params['v_max']
        self.v_min = params['v_min']
        self.u_max = params['u_max']
        self.u_min = params['u_min']
        self.du_max = params['du_max']
        self.du_min = params['du_min']

        # internal variables
        self.speed_intergal = 0.0

        # record
        self.t_traj = []
        self.u_traj = []
        self.target_traj = []

    def generate_control(self, ref_speed, obj_pred, pure_vel_keep=False):
        u_last = self.vehicle.u_last

        # integral term
        vel = self.vehicle.state[1]
        self.speed_intergal = self.speed_intergal + (ref_speed - vel) * self.dt
        # PI feedback
        targets = self._get_closest_dist(pred=obj_pred, pos_veh_front=self.vehicle.state[0] + self.vehicle.C2F)

        if self.verbose:
            print(self.t)
            print(targets)

        min_target = min(targets)

        # cruise control or obstacle avoidance
        if min_target < 1000 and not pure_vel_keep:
            d_dec = min_target - (self.vehicle.state[0] + self.vehicle.C2F) - self.d_safe
            if d_dec < 0:  # no enough dist. apply maximum break
                u = self.u_min
            else:
                u = - self.vehicle.state[1] ** 2 / (2 * d_dec)
        else:
            u = self.K_P_speed * (ref_speed - vel) + self.K_I_speed * self.speed_intergal

        # control rate constraint
        u = max(u, u_last + self.du_min * self.dt)
        u = min(u, u_last + self.du_max * self.dt)
        u = max(u, self.u_min)
        u = min(u, self.u_max)

        # record
        self.t_traj.append(self.t)
        self.u_traj.append(u)
        self.target_traj.append(targets)
        self.t = self.t + self.dt

        return u, None

    def _get_closest_dist(self, pred, pos_veh_front):
        closest_dist = np.full(len(pred), float('Inf'))
        if pos_veh_front < pred[0][0]: # before the vehicle reach the crossing point
            for i in range(len(pred)):
                lat_max = pred[i, 1] + pred[i, 2]
                lat_min = pred[i, 1] - pred[i, 2]
                if 0 < lat_min < self.W_lane or 0 < lat_max < self.W_lane or \
                        (0 < lat_min and lat_max < self.W_lane) or (lat_min < 0 and self.W_lane< lat_max):
                    lon_min = pred[i, 0] - pred[i, 2]
                    closest_dist[i] = lon_min

        return closest_dist



