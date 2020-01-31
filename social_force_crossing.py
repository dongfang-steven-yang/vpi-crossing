import numpy as np
import random
import math
from dynamics.dynamics import PointMassNewton
from interaction.social_force_yang import fun_decaying_exp, fun_des_vd
# ========================================================
# Motion Model: Social Force - New Version (Class)
# Created 2020-01-24


params_SocialForceCrossing = {
    'le': 0.2,
    'veh_A': 200.0,
    'veh_b': 2.6,
    'des_sigma': 1.0,
    'des_k': 300.0,
    'mu_gap': 4.0,
    'sigma_gap': 2.5,
    # todo: add different pedestrian type
    'mu_vd': 1.4,
    'sigma_vd': 0.2,

}


class SocialForceCrossing:
    def __init__(self, pedestrian, params, W_road, dt, t0, verbose=False):
        """

        :param pedestrians: a list of pedestrian objects
        :param params: common parameters
        :param dt:
        :param t0:
        """
        assert isinstance(pedestrian, PointMassNewton)
        self.ped = pedestrian
        self.params = params  # todo: in the future, some parameters have to be separated for individual peds
        self.dt = dt
        self.t = t0
        self.verbose = verbose

        # parameters
        self.le = params['le']
        self.veh_A = params['veh_A']
        self.veh_b = params['veh_b']
        self.des_sigma = params['des_sigma']
        self.des_k = params['des_k']
        self.W_lane = params['W_lane']

        # todo: make it more general - improve - modify
        self.y_wait = -0.5
        self.y_road_start = 0.0
        self.y_road_end = W_road

        # parameters of gap acceptance
        self.mu_gap = params['mu_gap']
        self.sigma_gap = params['sigma_gap']
        self.thr_gap = None

        # parameters of desired speed
        self.mu_vd = params['mu_vd']
        self.sigma_vd = params['sigma_vd']
        self.vd = None

        # variables
        self.state = None # states in {approach, wait, cross, finish}
        self.des = None

        # record
        self.t_traj = []
        self.f_total_traj = []
        self.fv_traj = []
        self.fd_traj = []
        self.state_traj = []
        self.des_traj = []
        self.gap_traj = []

        # initialize
        self.des = np.array([0, self.y_wait]).reshape(2, 1)  # assign destination
        self.vd = np.random.normal(self.mu_vd, self.sigma_vd)
        # self.vd = 1.1322

        self.ped.state[3] = self.vd
        self.thr_gap = np.random.normal(self.mu_gap, self.sigma_gap)
        # self.thr_gap = 4.2692

        self.state = 'approach' # state transition

    def transition(self, veh):
        # implementation
        if veh.state[1] == 0:
            gap = float('Inf')
        else:
            gap = (0 - (veh.state[0] + veh.C2F)) / veh.state[1]
        veh_passed = (veh.state[0] - veh.C2R) > self.ped.state[0]
        veh_blocked = (veh.state[0] - veh.C2R) < self.ped.state[0] < (veh.state[0] + veh.C2F)
        if self.state == 'approach':
            # action
            f_total, fd, fv = self.cal_forces(veh=None)
            # state transition
            if self.reachPoint(self.y_wait):
                self.state = 'wait'
        elif self.state == 'wait':
            # action
            f_total, fd, fv = self.cal_forces(veh=None)
            # state transition
            if (gap > self.thr_gap or veh_passed) and not veh_blocked:
                self.assign_des([0, 10])
                self.state = 'cross'
        elif self.state == 'cross':
            # action
            f_total, fd, fv = self.cal_forces(veh=veh)
            # state transition
            if self.reachPoint(self.y_road_end):
                # self.assign_des([0, 10]) # todo: use param
                self.state = 'finish'
        elif self.state == 'finish':
            f_total, fd, fv = self.cal_forces(veh=None)
        else:
            raise Exception('The pedestrian went to an unexpected state, please check.')

        # record
        self.t_traj.append(self.t)
        self.f_total_traj.append(f_total)
        self.fv_traj.append(fv)
        self.fd_traj.append(fd)
        self.state_traj.append(self.state)
        self.des_traj.append(self.des)
        self.gap_traj.append(gap)

        # update time
        self.t = self.t + self.dt
        return f_total, fd, fv, self.state

    def assign_des(self, des):
        self.des = np.array(des).reshape(2, 1)

    def reachPoint(self, y):
        return abs(y - self.ped.state[1]) < 0.5

    def cal_forces(self, veh=None):
        # # nearby peds
        # if surs is None:
        #     fr, fc, fn = np.array([[0], [0]]), np.array([[0], [0]]), np.array([[0], [0]])
        # else:
        #     raise Exception('Not implemented yet.')

        # the interacting veh
        if veh is None:
            fv, vd_adjusted = np.array([[0], [0]]), None
        else:
            fv, vd_adjusted = self._f_veh(veh)

        # the des force calculation must be called after the veh force cal. because of possible change of desired speed
        # des: if des is None, then estimate the des every time step
        if self.des is None:
            des = self._estimate_goal()
        else:
            des = self.des
        # increase desired speed if veh is ttc is too small
        if vd_adjusted is None:
            vd = self.vd
        else:
            vd = vd_adjusted
        fd = self._f_des(des, vd)

        # total force
        if fv[0][0] == float('Inf'): # collision
            return fv, fv, fv
        else:
            f_total = fd + fv
            return f_total, fd, fv

    def _estimate_goal(self):
        raise Exception('Implementation required.')
        pass

    def _f_veh(self, veh):
        # Boundary Force ------------------------------------------

        # contour
        x_v = veh.state[0]
        y_v = veh.lat_pos
        front = x_v + veh.C2F
        rear = x_v - veh.C2R
        left = y_v + veh.WIDTH / 2
        right = y_v - veh.WIDTH / 2

        # find influential point
        x_p = self.ped.state[0]
        y_p = self.ped.state[1]
        if x_p > front:
            if y_p > left:
                p_e = [front, left]
            elif y_p < right:
                p_e = [front, right]
            else:
                p_e = [front, y_p]
        elif front > x_p > rear:
            if y_p > left:
                p_e = [x_p, left]
            elif y_p < right:
                p_e = [x_p, right]
            else:
                p_e = [None, None]
        else:  # x < rear
            if y_p > left:
                p_e = [rear, left]
            elif y_p < right:
                p_e = [rear, right]
            else:
                p_e = [rear, y_p]

        # collision check
        if front + self.ped.R > x_p > rear - self.ped.R and left + self.ped.R > y_p > right - self.ped.R:
            return np.full((2, 1), float('Inf')), None  # collision, return inf value of fv

        # force magnitude
        p_p = np.array([x_p, y_p])
        p_e = np.array(p_e)
        d_v2p = np.linalg.norm(p_p - p_e)
        v_v2p = (p_p - p_e) / d_v2p
        v_v2p = v_v2p.reshape(2, 1)

        fv = fun_decaying_exp(d_v2p - self.le - self.ped.R, self.veh_A, self.veh_b) * v_v2p

        # Adjusting desired speed ---------------------------------
        d_lon = x_p - self.ped.R - front  # longitudinal dist from ped to veh
        d_rem = self.W_lane + self.ped.R - y_p  # remaining distance
        speed_v = veh.state[1]
        if speed_v == 0: # time to collision for vehicle
            ttc = float('Inf')
        else:
            ttc = d_lon / speed_v
        ttf = d_rem / self.vd  # time to clear the crosswalk for pedestrian
        if 0 < ttc < ttf:
            vd_adjusted = d_rem / ttc
        else:
            vd_adjusted = None

        return fv, vd_adjusted

    def _f_des(self, des, v0):
        vec_vd = fun_des_vd(self.ped.state[:2], des, v0, self.des_sigma)
        fd = self.des_k * (vec_vd - self.ped.state[2:].reshape(2, 1))
        return fd


# TEST
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 5))

    # vehicle force
    paras = params_SocialForceCrossing
    ped_R = 0.27
    ds = np.arange(0, 10, 0.1)
    fvs_mag = []
    for d in ds:
        # fvs_mag.append(fun_decaying_exp(d - paras['le'] - ped_R, paras['veh_A'], paras['veh_b']))
        fvs_mag.append(fun_decaying_exp(d, paras['veh_A'], paras['veh_b']))

    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(ds, fvs_mag)
    ax1.set_xlabel('distance')
    ax1.set_ylabel('magnitude')
    ax1.grid()
    plt.show()

    # destination
    d_range = np.arange(0, 10, 0.1)
    vds_mag = np.empty(len(d_range))
    for i, d in enumerate(d_range):
        vec_vd = fun_des_vd(np.array([[0], [0]]),
                            np.array([[d], [0]]),
                            1.4,
                            paras['des_sigma'])
        vds_mag[i] = np.linalg.norm(vec_vd)

    ax = fig.add_subplot(1,2,2)
    ax.plot(d_range, vds_mag)
    ax.grid(linestyle='--', linewidth=0.5)
    ax.title.set_text('desired speed magnitude')
    ax.set_xlabel('distance to goal')
    ax.set_ylabel('desired speed magnitude')



