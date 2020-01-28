import numpy as np
import cvxpy as cp


class LongitudinalMPC:
    def __init__(self, vehicle, N_pred=15,
                 v_max=22.5, v_min=0,
                 a_max=7, a_min=-7,
                 da_max=5, da_min=-5,
                 d_safe=5, verbose=False):
        self.vehicle = vehicle
        self.dt = vehicle.dt
        self.verbose = verbose

        #
        self.n_state = vehicle.A.shape[1]
        self.n_control = vehicle.B.shape[1]
        self.N = N_pred  # prediction horizon
        self.d_safe = d_safe # todo @ improve, use TTC

        # constraints
        self.v_max = v_max
        self.v_min = v_min
        self.a_max = a_max
        self.a_min = a_min
        self.da_max = da_max
        self.da_min = da_min

        # cost weights
        self.W_state = 5.0
        self.W_control = 1.0

    def generate_control(self, ref_speed, obj_pred, u_last):
        # initial condition
        x_0 = self.vehicle.state.reshape(2)

        # data
        closest_dist = self._get_closest_dist(obj_pred)
        if self.verbose:
            print('Closest Dist:')
            print(closest_dist)

        # variables
        x = cp.Variable((self.n_state, self.N+1))
        u = cp.Variable((self.n_control, self.N))

        # cost and constraints
        cost = 0.0
        constr = []

        for t in range(self.N):
            cost += self.W_state * cp.sum_squares(x[1, t+1] - ref_speed) + self.W_control * cp.sum_squares(u[:, t])
            constr += [
                x[:, t + 1] == self.vehicle.A@x[:, t] + self.vehicle.B@u[:, t],  # dynamics
                x[1, t + 1] <= self.v_max,
                x[1, t + 1] >= self.v_min, # vel constraint
                u[:, t] <= self.a_max,
                u[:, t] >= self.a_min,  # acc constraint
                closest_dist[t+1] - x[0, t+1] >= self.d_safe  # safe distance
            ]
            # acc rate constraint
            if t > 0:
                constr += [u[:, t] - u[:, t - 1] >= self.da_min * self.dt,
                           u[:, t] - u[:, t - 1] <= self.da_max * self.dt]
            else:
                constr += [u[:, t] - u_last >= self.da_min * self.dt,
                           u[:, t] - u_last <= self.da_max * self.dt]

        # initial condition
        constr += [x[:, 0] == x_0]

        # terminal condition
        constr += [closest_dist[self.N] - x[0, self.N] >= (x[1, self.N] / 2) * self.v_max / (- self.a_min) + self.d_safe]
        #     constr += [x[1], self.N] == 0

        # construct problem
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        # problem.solve(solver=cp.ECOS,
        #               verbose=False)
        # debug
        if self.verbose:
            print('X=')
            print(x.value)
            print('U=')
            print(u.value)
        # self.vehicle.A.dot(x_0.reshape(2, 1)) + self.vehicle.B.dot(np.array(1))

        # first control
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            action = u[0, 0].value
            print(f'MPC found solution u = {action}')
            return action

        else:
            print(f'MPC infeasible')
            # test
            print('------------------------------------')
            print('Test Emergency Braking:')
            xx = x_0.reshape(2, 1)
            uu = u_last
            for i in range(20):
                uu = max(uu - 0.5, -7.0)
                print(f'u = {uu}')
                xx = self.vehicle.A.dot(xx) + self.vehicle.B.dot(np.array(uu).reshape(1, 1))
                print(f'x = {xx.reshape(2)}')
                print(f'at step {i+1} dist to predicted obs = {closest_dist[i+1]:.4f} - {xx[0][0]:.4f} = {closest_dist[i+1] - xx[0][0]:.4f}')

            return None
            # raise Exception('MPC infeasible')

    def _get_closest_dist(self, predicted_interval):
        closest_dist = 1000 * np.ones(self.N+1)
        if predicted_interval is None:
            pass
        else:
            reachable = predicted_interval[3]  # obtain reachable set
            for i in range(self.N+1):
                # check lateral distance
                lat_max = reachable[1, 2*i] + reachable[1, 2*i+1]
                lat_min = reachable[1, 2*i] - reachable[1, 2*i+1]
                if 0 < lat_min < 3 or 0 < lat_max < 3 or (0 < lat_min and lat_max < 3) or (0 > lat_min and lat_max > 3):
                    # obstacle inside ego lane
                    lon_min = reachable[0, 2*i] - reachable[0, 2*i+1]
                    closest_dist[i] = lon_min

        return closest_dist




