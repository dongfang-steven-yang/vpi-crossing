import numpy as np
from pathlib import Path

from dynamics import PointMassNewton, DynamicLongitudinal
from dynamics import PointMassNewton_params_ped as params_ped
from dynamics import DynamicLongitudinal_params_simple as params_veh, KinematicBicycle_params_GAC_GE3

from controller import ModelPredictiveController, params_control_mpc, params_control_general
from controller import PIDController, params_control_pid

# from prediction.reachable_set import PointMassReachableSet
from predictor import PredictorLinear

from social_force_crossing import SocialForceCrossing, params_SocialForceCrossing

import pickle

from visualizer import SimVisCrossing


# paths
result_path = 'result'

# Hyper-parameters ---------------------------------------------------------------------------------------------------
DT = 0.1  # sec
T0 = 0.0
T_sim_total = 10.0
VERBOSE = False

# Parameters ---------------------------------------------------------------------------------------------------------

# facts
R_MPH2mps = 2.237 # ratio from MPH to m/s

# road lines
W_lane = 3.2 # m, from https://safety.fhwa.dot.gov/geometric/pubs/mitigationstrategies/chapter3/3_lanewidth.cfm
X_min = -130.0
X_max = 30.0

# Prediction Horizon
T_prediction = 3.0  # sec, time horizon of prediction


# Functions ----------------------------------------------------------------------------------------------------------


def sim_once(init_state, control_method, predict_method, num, date_time, suppress_video_save=False):
    """

    :param init_state:
    :param control_method:
    :param predict_method:
    :param num:
    :param date_time:
    :param suppress_video_save: used for randomly saving videos when runnning a lot of simulations.
    :return:
    """
    pickle_path = Path(
        result_path,
        f'{date_time}_sim_result_{control_method}_{predict_method}',
        f'pos {int(init_state[0]):3d} vel {int(init_state[1]):2d} itr {num:3d}.p'
    )
    Path(pickle_path).parent.mkdir(parents=True, exist_ok=True)

    # Configs
    np.set_printoptions(precision=4, suppress=True)

    # Initialize the Scene -------------------------------------------------------------------------------------------

    # road lines
    layout = {
        'lane_line_solid': [np.array([[X_min, X_max], [0, 0]]),
                            np.array([[X_min, X_max], [W_lane * 2, W_lane * 2]])],
        'lane_line_dashed': [np.array([[X_min, X_max], [W_lane, W_lane]])]
    }

    # pedestrian dynamics
    # params_ped.update({'W_lane': W_lane})
    ped = PointMassNewton(params=params_ped, initial_state=[0.0, -2.0, 0.0, 1.0], dt=DT, t0=T0)

    # pedestrian interaction - motion model
    params_SocialForceCrossing.update({'W_lane': W_lane})
    ped_sfm = SocialForceCrossing(pedestrian=ped, params=params_SocialForceCrossing, W_road=2*W_lane, dt=DT, t0=T0)

    # vehicle dynamics
    params_veh.update(KinematicBicycle_params_GAC_GE3)
    params_control_general.update({'W_lane': W_lane})
    params_veh.update(params_control_general)
    veh_ego = DynamicLongitudinal(params=params_veh, initial_state=init_state, dt=DT, t0=T0, lat_pos=W_lane / 2, verbose=False)

    # vehicle interaction - controller
    if control_method == 'mpc':
        # -- MPC
        params_control_mpc.update(params_control_general)
        params_control_mpc['N_pred'] = round(T_prediction / DT)
        con_ego = ModelPredictiveController(vehicle=veh_ego, params=params_control_mpc, verbose=True)
    elif control_method == 'pid':
        # -- PID
        params_control_pid.update(params_control_general)
        con_ego = PIDController(vehicle=veh_ego, params=params_control_pid, verbose=False)
    else:
        raise Exception('Invalid Controller')

    # vehicle interaction - predictor
    if predict_method == 'reachable':
        # -- Reachable Set
        # predictor = PointMassReachableSet(dt=DT)
        pass
    elif predict_method == 'lin_last_obs':
        # -- Linear
        predictor = PredictorLinear(mode='last_obs', dt=DT, t0=T0)
    else:
        raise Exception('Invalid Predictor')

    # Simulation -----------------------------------------------------------------------------------------------------

    for i in range(int(np.floor(T_sim_total / DT))):
        # loop info
        if VERBOSE:
            print('====================================================')
            print(f'simulation running at t = %.3f' % (i * DT + T0))

        # pedestrian motion
        f_total, fd, fv, ped_state = ped_sfm.transition(veh=veh_ego)
        if VERBOSE:
            print(f'Social Force: state = {ped_state}, f_total = {fd.reshape(2)}(fd) + {fv.reshape(2)}(fv) = {f_total.reshape(2)}.')

        # prediction
        # prediction = predictor.predict(state_now=ped.state, acc_limit=LIMIT_acc, t_pred=T_pred_mpc, radius=ped.R)
        prediction = predictor.predict(traj_past=ped.state.reshape(1, 4), t_pred=T_prediction, radius=ped.R)

        # planning and control
        u, feasible = con_ego.generate_control(ref_speed=init_state[1], obj_pred=prediction)
        # u, feasible = con_ego.generate_control(ref_speed=init_state[1], obj_pred=prediction, pure_vel_keep=True)

        if VERBOSE:
            print(f'Applied u = {u:.4f}.')

        # state update
        ped.update(f_total)
        veh_ego.update(u=u)

        # if collision, write collision info, terminate the loop
        if f_total[0][0] == float('Inf'):
            print('--------------------> Collision!')
            file_er = open(pickle_path[:-2] + '_collision_info.txt', 'w')
            file_er.write(f'collision happened at t = {i * DT + T0:.3f}.')
            break

    # Save Data -----------------------------------------------------------------------------------------------------
    pickle.dump((layout, ped, veh_ego, ped_sfm, predictor, con_ego), open(pickle_path, "wb"))

    # Visualization -------------------------------------------------------------------------------------------------
    if np.random.uniform() < 0.01 and not suppress_video_save:
        print('Saving video. This may take some time, please wait.')
        vis = SimVisCrossing(layout=layout, ped=ped, veh=veh_ego, sfm=ped_sfm, predictor=predictor, con=con_ego)
        vis.animate(save_video=True, save_path=pickle_path[:-1] + 'mp4')


def playback(init_state, control_method, predict_method, num, date_time):
    # pickle_path = f'vpi_crossing_result/{date_time}_sim_result_{control_method}_{predict_method}/' \
    #               f'pos {int(init_state[0]):3d} ' \
    #               f'vel {int(init_state[1]):2d} ' \
    #               f'itr {num:3d}.p'

    pickle_path = Path(
        result_path,
        f'{date_time}_sim_result_{control_method}_{predict_method}',
        f'pos {int(init_state[0]):3d} vel {int(init_state[1]):2d} itr {num:3d}.p'
    )

    layout, ped, veh_ego, ped_sfm, predictor, con_ego = pickle.load(open(pickle_path, "rb"))
    print('Saving video. This may take some time, please wait.')
    vis = SimVisCrossing(layout=layout, ped=ped, veh=veh_ego, sfm=ped_sfm, predictor=predictor, con=con_ego)
    vis.animate(save_video=True, save_path=str(pickle_path)[:-1] + 'mp4')


def display_results(init_state, control_method, predict_method, num, date_time):
    # pickle_path = f'vpi_crossing_result/{date_time}_sim_result_{control_method}_{predict_method}/' \
    #               f'pos {int(init_state[0]):3d} ' \
    #               f'vel {int(init_state[1]):2d} ' \
    #               f'itr {num:3d}.p'

    pickle_path = Path(
        result_path,
        f'{date_time}_sim_result_{control_method}_{predict_method}',
        f'pos {int(init_state[0]):3d} vel {int(init_state[1]):2d} itr {num:3d}.p'
    )

    layout, ped, veh_ego, ped_sfm, predictor, con_ego = pickle.load(open(pickle_path, "rb"))
    print('Creating figures ...')
    vis = SimVisCrossing(layout=layout, ped=ped, veh=veh_ego, sfm=ped_sfm, predictor=predictor, con=con_ego)
    vis.plot(save_figure=True, save_path=str(pickle_path)[:-1] + 'png')


if __name__ == '__main__':
    # init_states = [[-20, 8], [-20, 6], [-20, 4], [-20, 2], [-25, 8], [-25, 6], [-25, 4]]
    init_states = [[-25, 8]]
    method = 'pid'
    note = 'test05example'
    num = 3

    for init_state in init_states:
        print(f'sim with initial state = {init_state}')
        sim_once(
            init_state=init_state,
            control_method=method,
            predict_method='lin_last_obs',
            num=num,
            date_time=note,
            suppress_video_save=True
        )

        playback(
            init_state=init_state,
            control_method=method,
            predict_method='lin_last_obs',
            num=num,
            date_time=note
        )

        display_results(
            init_state=init_state,
            control_method=method,
            predict_method='lin_last_obs',
            num=num,
            date_time=note
        )

