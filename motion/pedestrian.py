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


