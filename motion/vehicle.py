import numpy as np


# ==================================================================
# Vehicle Dynamics for Longitudinal-only Motion

DynamicLongitudinal_params_simple = {
    'M': 2000,  # kg, weight of the vehicle, Volvo XC90 weighs 1993 kg
    'alpha': 100,  # friction
    'LENGTH': 4.5,  # length of the vehicle contour
    'WIDTH': 2.0,  # width of the vehicle contour
    'BACKTOWHEEL': 1.0,  # dist from backend to rear wheel
    'WHEEL_LEN': 0.3,  # wheel length
    'WHEEL_WIDTH': 0.2,  # wheel width
    # make sure WB = lr + lf
    'WB': 2.5,  # wheelbase, dist from rear wheel to front wheel
    'LR': 0.0,  # dist from rear wheel to (defined) center
    'LF': 2.5  # dist from front wheel to (defined) center
}

# KinematicBicycle_params_GAC_GE3 = {
#     'LENGTH': 4.5,  # length of the vehicle contour
#     'WIDTH': 2.0,  # width of the vehicle contour
#     'BACKTOWHEEL': 1.0,  # dist from backend to rear wheel
#     'WHEEL_LEN': 0.3,  # wheel length
#     'WHEEL_WIDTH': 0.2,  # wheel width
#     # make sure WB = lr + lf
#     'WB': 2.5,  # wheelbase, dist from rear wheel to front wheel
#     'LR': 0.0,  # dist from rear wheel to (defined) center
#     'LF': 2.5  # dist from front wheel to (defined) center
# }


class DynamicLongitudinal:
    def __init__(self, params, initial_state, dt, t0, lat_pos, verbose=False):
        # shape - used for visualization
        self.LENGTH = params['LENGTH']  # [m]
        self.WIDTH = params['WIDTH']  # [m]
        self.BACKTOWHEEL = params['BACKTOWHEEL']  # [m]
        self.WHEEL_LEN = params['WHEEL_LEN']  # [m]
        self.WHEEL_WIDTH = params['WHEEL_WIDTH']  # [m]
        self.WB = params['WB']  # [m]
        self.LR = params['LR']
        self.LF = params['LF']
        assert self.LR + self.LF == self.WB
        self.C2R = self.LR + self.BACKTOWHEEL
        self.C2F = self.LENGTH - self.C2R

        # general parameters
        self.dt = dt
        self.verbose = verbose

        # model parameters
        self.M = params['M']
        self.alpha = params['alpha']

        # state constraint
        self.v_max = params['v_max']
        self.v_min = params['v_min']

        # transition matrix
        self.A = np.array([[1, dt], [0, 1-self.alpha*dt/self.M]])
        self.B = np.array([[0], [dt]])

        # state variables
        self.t = t0
        self.u_last = 0.0
        self.state = np.array(initial_state).reshape(2)
        self.lat_pos = lat_pos

        # record
        self.t_traj = [t0]
        self.state_traj = [np.array([self.state[0], self.lat_pos, 0.0, self.state[1]])]
        self.action_traj = []

    def update(self, u):
        # speed constraint
        u_received = u
        u_lower = self.v_min/self.dt + self.alpha/self.M*self.state[1] - self.state[1]/self.dt
        u_upper = self.v_max/self.dt + self.alpha/self.M*self.state[1] - self.state[1]/self.dt
        u = max(u, u_lower)
        u = min(u, u_upper)
        if self.verbose:
            print(f' -- veh pos={self.state[0]:.4f}, speed={self.state[1]:.4f}')
            print(f' -- received u={u_received:.4f}, executed u={u:.4f}')
            print(f' -- dynamics-based u_bounds=[{u_lower:.4f},{u_upper:.4f}]')

        u = np.array(u).reshape(1, 1)
        state_new = self.A.dot(self.state.reshape(2, 1)) + self.B.dot(u)
        self.state = state_new.reshape(2)
        self.u_last = u[0][0]
        self.t = self.t + self.dt

        # record trajectory
        self.state_traj.append(np.array([self.state[0], self.lat_pos, 0.0, self.state[1]]))
        self.action_traj.append(u.reshape(1))
        self.t_traj.append(self.t)
