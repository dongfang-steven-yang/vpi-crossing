import numpy as np
import random
from pathlib import Path

from dynamics import PointMassNewton, PointMassNewton_params_ped as params_PointMassNewton_ped
from dynamics import DynamicLongitudinal, \
    DynamicLongitudinal_params_simple as params_DynamicLongitudinal, \
    KinematicBicycle_params_GAC_GE3 as params_KinematicBicycle_GAC_GE3

from ped.social_force_vci.social_force_yang import calculate_total_force, params_SocialForceYang as params_vci

from uber_case.longitudinal_mpc import LongitudinalMPC
from veh.modular_pipline.planning.splines.cubic_spline import generate_trajectory

from uber_case_legacy.tracker import TrackerSingle
from uber_case_legacy.predictor import PredictorReachableSet
from sim_env.visualizers import SimResultVisualizer

# Hyper-parameters ---------------------------------------------------------------------------------------------------
sensor_noise = True  # todo @ future

# sensor_mode = 'simulated'
sensor_mode = 'original'

discard_tracking = False
# discard_tracking = True

# predictor_mode = 'uber_mode'
# predictor_mode = 'linear_vel'
predictor_mode = 'reach_acc'

VIDEO_save = True
VIDEO_filename = sensor_mode \
                 + '_' + ('drop_tracking' if discard_tracking else 'keep_tracking') \
                 + '_' + predictor_mode

# Parameters ---------------------------------------------------------------------------------------------------------
# params for sim
DT = 0.1 # sec
T0 = -6.0
T_sim = 10 # todo social_force_fundamental time here
sigma = 0.001

W_lane = 3.2 # m, from https://safety.fhwa.dot.gov/geometric/pubs/mitigationstrategies/chapter3/3_lanewidth.cfm
X_min = -130
X_max = 30

R_MPH2mps = 2.237 # ratio from MPH to m/s

# dict for classification
SENSOR_class = {
    0: 'unknown',
    1: 'vehicle',
    2: 'cyclist',
    3: 'pedestrian'
}

# sensor - classification weights for social_force_fundamental
SENSOR_hold_step = 3
WEIGHTS_far = [0.5, 0.5, 0, 0]
WEIGHTS_close = [0.2, 0.0, 0.8, 0]

# path predictor
LIMIT_acc = 0.6  # m/s^2
T_pred = 2.0  # sec

# controller - MPC
v_max = 22.5
v_min = 0
a_max = 7
a_min = -7
da_max = 5
da_min = -5
d_safe = 5


# Global Variables ---------------------------------------------------------------------------------------------------
sensor_continually_tracked = 0
sensor_last_classification = 0

# Functions ----------------------------------------------------------------------------------------------------------


def get_social_froce_params():
    params = {}
    # todo @ improve future
    params.update(params_KinematicBicycle_GAC_GE3)
    params.update(params_PointMassNewton_ped)
    params.update(params_vci)
    return params


def get_veh_ref_traj():
    ref_vel = 45 / R_MPH2mps
    ref_start = [X_min, W_lane / 2, 0, ref_vel]
    ref_end = [X_max, W_lane / 2, 0, ref_vel]
    ref_traj = generate_trajectory([ref_start[0], ref_end[0]], [ref_start[1], ref_end[1]], ref_vel)
    return ref_traj


def initialize_scene(t0):
    layout = {
        'lane_line_solid': [np.array([[X_min, X_max], [0, 0]]),
                            np.array([[X_min, X_max], [W_lane * 2, W_lane * 2]])],
        'lane_line_dashed': [np.array([[X_min, X_max], [W_lane, W_lane]])]

    }

    # pedestrian - social force model
    s0_ped = [0, W_lane * 3, 0.1, -0.9]
    ped = PointMassNewton(params=params_PointMassNewton_ped, initial_state=s0_ped, dt=DT, t0=t0)

    # vehicle - bicycle model
    # s0_veh = [-44 / R_MPH2mps * 6.0, W_lane / 2, 0, 45 / R_MPH2mps]
    # veh = KinematicBicycle(params=params_KinematicBicycle_GAC_GE3, initial_state=s0_veh, dt=DT, t0=t0)
    s0_veh = [-44 / R_MPH2mps * 6.0, 45 / R_MPH2mps]
    params_DynamicLongitudinal.update(params_KinematicBicycle_GAC_GE3)
    veh = DynamicLongitudinal(params=params_DynamicLongitudinal, initial_state=s0_veh, dt=DT, t0=t0, lat_pos=W_lane/2)

    # controller
    # con = PurePersuitLocal(vehicle=veh, direct_assign=False)
    # con.update_path(traj=get_veh_ref_traj())
    con = LongitudinalMPC(vehicle=veh, N_pred=round(T_pred/DT),
                          v_max=v_max, v_min=v_min,
                          a_max=a_max, a_min=a_min,
                          da_max=da_max, da_min=da_min,
                          d_safe=d_safe
                          )

    # tracker
    tracker = TrackerSingle(dt=DT, t0=t0)

    # predictor
    predictor = PredictorReachableSet(dt=DT, acc_limit=LIMIT_acc, t_pred=T_pred, ped_R=ped.R)

    return layout, ped, veh, con, tracker, predictor


def sensor_original(t):
    if t + sigma < - 5.6:
        obj_class = None
    elif - 5.6 < t + sigma < - 5.2:
        obj_class = 1
    elif - 5.2 < t + sigma < - 4.2:
        obj_class = 0
    elif - 4.2 < t + sigma < - 3.8:
        obj_class = 1
    elif - 3.8 < t + sigma < - 2.6:
        obj_class = -1 # alternating
    elif - 2.6 < t + sigma < - 1.5:
        obj_class = 2
    elif - 1.5 < t + sigma < - 1.2:
        obj_class = 0
    elif - 1.2 < t + sigma < 1.0:
        obj_class = 2
    else:
        obj_class = 2
    return obj_class


def sensor_detect(ped):
    global sensor_continually_tracked, sensor_last_classification

    if ped.t + sigma < -5.6:
        obj_pos = None
        obj_class = None

    else:
        # position -----------------------------------------------------
        obj_pos = ped.state[:2]

        # classification ------------------------------------------------
        if sensor_mode is 'original':
            obj_class = sensor_original(ped.t)
            if obj_class == -1: # alternating
                obj_class = random.choices(population=[0, 1, 2, 3], weights=WEIGHTS_far)[0]

        elif sensor_mode is 'simulated':
            # single step classify
            if ped.t + sigma < -2.6:
                obj_class = random.choices(population=[0, 1, 2, 3], weights=WEIGHTS_far)[0]
            else:
                obj_class = random.choices(population=[0, 1, 2, 3], weights=WEIGHTS_close)[0]
            # hold classification result
            if sensor_continually_tracked < 3:
                obj_class = sensor_last_classification
                sensor_continually_tracked = sensor_continually_tracked + 1
            else:
                if obj_class == sensor_last_classification:
                    sensor_continually_tracked = sensor_continually_tracked + 1
                else:
                    sensor_continually_tracked = 0
        else:
            obj_class = None
            raise Exception('Invalid sensor mode !')

    # store last
    sensor_last_classification = obj_class

    return obj_class, obj_pos


def main():
    np.set_printoptions(precision=4, suppress=True)

    params_sfm = get_social_froce_params()
    layout, ped, veh, con, tracker, predictor = initialize_scene(t0=T0)
    ped_des = [3, -5]
    perception_result = [] # [t_now, traj_class_backward, traj_pos_backward, prediction]

    # social_force_fundamental
    for i in range(int(np.floor(T_sim / DT))):
        # loop info
        print('now t = %.3f' % (i * DT + T0))

        # perception -------------------------------------------------------------------------------------------
        obj_class, obj_pos = sensor_detect(ped)
        tracker_active = tracker.update(obj_class=obj_class, obj_pos=obj_pos)

        # prediction --------------------------------------------------------------------------------------
        if tracker_active:
            traj_class_backward, traj_pos_backward, t_now = tracker.get_tracking(discard_tracking=discard_tracking)
            reachable_set = predictor.predict(mode=predictor_mode,
                                              traj_pos_backward=traj_pos_backward,
                                              traj_class_backward=traj_class_backward)
            perception_result.append([t_now, traj_class_backward, traj_pos_backward, reachable_set])
        else:
            perception_result.append(None)

        # planning and control --------------------------------------------------------------------------------
        # pure pursuit steering
        # acc, steer, _, _, _ = con.generate_control()
        # u = (acc, steer)

        # # longitudinal MPC
        if i == 0:
            u_last = 0
        else:
            u_last = veh.action_traj[-1][0]
        u = con.generate_control(ref_speed=45/R_MPH2mps, obj_pred=perception_result[i], u_last=u_last)
        if u is None:
            print('Apply emergency breaking')
            u = max(a_min, u_last + da_min * DT)
        print(f'executed u = {u}')

        # # temp todo @ remove
        # u = 2*np.random.rand(1)[0]

        # social_force_fundamental update ------------------------------------------------------------------------------------------
        # update ped
        F_sfm, _, _, _, _, _  = calculate_total_force(params=params_sfm, ego=ped.state, surs=[], vehs=[], env=[], des=ped_des, dt=DT)
        ped.update(force=F_sfm)
        # update veh
        veh.update(u=u)

    # visualization
    vis = SimResultVisualizer(peds=[ped], vehs=[veh], layout=layout, prediction=perception_result, dt=DT)
    vis.animate(save_video=VIDEO_save, result_path=Path('./result', VIDEO_filename + '.mp4'))


if __name__ == '__main__':
    main()