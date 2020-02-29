import numpy as np

# ==================================================================
# Point-Mass Newton

PointMassNewton_params_ped = {
    'm': 80,  # mass of the point
    'R': 0.27,  # virtual boundary
    'v_max': 2.5,
    'a_max': 5
}


class PointMassNewton:
    """
    The Point Mass Newton can be applied as pedestrian dynamics, or other small agents which has similar properties.
    """
    def __init__(self, params, initial_state, dt, t0, int_method='SmartEuler'):

        # parameters
        self.R = params['R']
        self.m = params['m']
        self.v_max = params['v_max']
        self.a_max = params['a_max']
        self.dt = dt

        # state
        self.state = np.array(initial_state).reshape(4)  # x, y, vx, vy
        self.t = t0

        # trajectories
        self.t_traj = [t0]
        self.state_traj = [self.state.reshape(4)]  # list of [x, y, vx, vy]
        self.action_traj = []  # list of [F_x, F_y]

        # point-mass model integration method, see the following page:
        # https://mathsimulationtechnology.wordpress.com/2012/01/28/newtons-laws-of-motion/
        if int_method == 'SmartEuler':
            self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
            self.B = np.array([[dt**2/(self.m*2), 0], [0, dt**2/(self.m*2)], [dt/self.m, 0], [0, dt/self.m]])
            self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        elif int_method == 'ForwardEuler':
            self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
            self.B = np.array([[0, 0], [0, 0], [dt/self.m, 0], [0, dt/self.m]])
            self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        else:
            raise Exception("invalid integration method for pedestrian Newtownian dynamics")

    def update(self, force):
        """

        :param force: a column vector of size 2, numpy array
        :return:
        """
        # force constraint
        force = np.array(force).reshape(2, 1)
        if np.linalg.norm(force) > self.m * self.a_max:
            force = force / np.linalg.norm(force) * self.m * self.a_max
        v_now = self.state[2:].reshape(2,1)
        v_new = v_now + self.dt * force / self.m
        if np.linalg.norm(v_new) > self.v_max:
            force = self.m / self.dt * (v_new / np.linalg.norm(v_new) * self.v_max - v_now)

        # update state
        state_new = self.A.dot(self.state.reshape(4, 1)) + self.B.dot(force)
        self.state = state_new.reshape(4)
        self.t = self.t + self.dt

        # record trajectory
        self.state_traj.append(self.state.reshape(4))
        self.action_traj.append(force.reshape(2))
        self.t_traj.append(self.t)

        return self.state


# ==================================================================
# Vehicle Dynamics for Longitudinal-only Motion

DynamicLongitudinal_params_simple = {
    'M': 2000,  # kg, weight of the vehicle, Volvo XC90 weighs 1993 kg
    'alpha': 100,  # friction
}


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


# ==================================================================
# Kinematic Bicycle Model

KinematicBicycle_params_GAC_GE3 = {
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

KinematicBicycle_params_EZGO_golf = {
    'LENGTH': 2.2,  # length of the vehicle contour
    'WIDTH': 1.2,  # width of the vehicle contour
    'BACKTOWHEEL': 0.6,  # dist from backend to rear wheel
    'WHEEL_LEN': 0.3,  # wheel length todo @ check
    'WHEEL_WIDTH': 0.2,  # wheel width
    # make sure WB = lr + lf
    'WB': 1.4,  # wheelbase, dist from rear wheel to front wheel
    'LR': 0.4,  # dist from rear wheel to (defined) center
    'LF': 1.0  # dist from front wheel to (defined) center
}


class KinematicBicycle:
    def __init__(self, params, initial_state, dt, t0):
        # parameters
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
        self.dt = dt

        # initial state
        self.state = np.array(initial_state).reshape(4) # x, y, yaw, vel
        self.t = t0

        # trajectories
        self.t_traj = [t0]
        self.state_traj = [self.state]
        self.action_traj = []

    def update(self, u):

        # update state
        u = np.array(u).reshape(2)
        acc = u[0]   # gas/brake
        delta = u[1]  # steering

        x = self.state[0]
        y = self.state[1]
        psi = self.state[2]
        vel = self.state[3]
        beta = np.arctan(self.LR * np.tan(delta) / (self.LF + self.LR))

        x_new = np.empty(4)
        x_new[0] = x + self.dt * vel * np.cos(psi + beta)
        x_new[1] = y + self.dt * vel * np.sin(psi + beta)
        x_new[2] = psi + self.dt * vel * np.cos(beta) / (self.LF + self.LR) * np.tan(delta)
        x_new[2] = _yaw_angle_correction(x_new[2])
        x_new[3] = vel + self.dt * acc

        self.state = x_new
        self.t = self.t + self.dt

        # record trajectory
        self.state_traj.append(self.state)
        self.action_traj.append(u)
        self.t_traj.append(self.t)

        return x_new

    def cal_curvature_traj(self):

        x = np.array(self.state_traj)[:, 0]
        y = np.array(self.state_traj)[:, 1]
        yaw = np.array(self.state_traj)[:, 2]

        curvature = []
        for i in range(len(yaw)-1):
            yaw_d = (float(yaw[i + 1] - yaw[i]) + np.pi) % (2.0 * np.pi) - np.pi
            curvature.append(yaw_d / np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2))

        return curvature


# Methods used within dynamics module:
def _yaw_angle_correction(theta):
    """
    correct yaw angle so that it always remains in [-pi, pi]
    :param theta:
    :return:
    """
    theta_corrected = (theta + np.pi) % (2.0 * np.pi) - np.pi
    return theta_corrected
