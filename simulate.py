import numpy as np
import pickle
from pathlib import Path
from perception.vehicle import PredictorLinear
from interaction.vehicle import ModelPredictiveController, params_control_mpc, params_control_general
from interaction.vehicle import PIDController, params_control_pid
from interaction.pedestrian import SocialForceCrossing, params_SocialForceCrossing
from motion.pedestrian import PointMassNewton, PointMassNewton_params_ped as params_ped
from motion.vehicle import DynamicLongitudinal, DynamicLongitudinal_params_simple as params_veh
np.set_printoptions(precision=4, suppress=True)

# Parameters ---------------------------------------------------------------------------------------------------------
# sim info
DT = 0.1  # sec
T0 = 0.0
T_sim_total = 10.0

# road info
W_lane = 3.2  # m, from https://safety.fhwa.dot.gov/geometric/pubs/mitigationstrategies/chapter3/3_lanewidth.cfm
X_min = -130.0
X_max = 30.0
R_MPH2mps = 2.237  # ratio converting MPH to m/s

# controller
T_prediction = 3.0  # sec, time horizon of prediction

# saving results
result_path = 'results'
Path(result_path).mkdir(parents=True, exist_ok=True)


def simulate(init_state, control_method, verbose=False):

    # Initialize the Scene -------------------------------------------------------------------------------------------
    # road layout
    layout = {
        'lane_line_solid': [np.array([[X_min, X_max], [0, 0]]),
                            np.array([[X_min, X_max], [W_lane * 2, W_lane * 2]])],
        'lane_line_dashed': [np.array([[X_min, X_max], [W_lane, W_lane]])]
    }

    # pedestrian dynamics
    ped = PointMassNewton(params=params_ped, initial_state=[0.0, -2.0, 0.0, 1.0], dt=DT, t0=T0)

    # pedestrian interaction - motion model
    params_SocialForceCrossing.update({'W_lane': W_lane})
    ped_sfm = SocialForceCrossing(pedestrian=ped, params=params_SocialForceCrossing, W_road=2*W_lane, dt=DT, t0=T0)

    # vehicle dynamics
    params_control_general.update({'W_lane': W_lane})
    params_veh.update(params_control_general)
    veh_ego = DynamicLongitudinal(params=params_veh, initial_state=init_state, dt=DT, t0=T0, lat_pos=W_lane / 2, verbose=False)

    # vehicle interaction - controller
    if control_method == 'mpc':
        # model predictive control
        params_control_mpc.update(params_control_general)
        params_control_mpc['N_pred'] = round(T_prediction / DT)
        con_ego = ModelPredictiveController(vehicle=veh_ego, params=params_control_mpc, verbose=True)
    elif control_method == 'oac':
        # obstacle avoidance control
        params_control_pid.update(params_control_general)
        con_ego = PIDController(vehicle=veh_ego, params=params_control_pid, verbose=False)
    elif control_method == 'vkc':
        # velocity keeping control
        params_control_pid.update(params_control_general)
        con_ego = PIDController(vehicle=veh_ego, params=params_control_pid, pure_vel_keep=True, verbose=False)
    else:
        raise Exception('Invalid Controller Type')

    # vehicle interaction - predictor
    predictor = PredictorLinear(dt=DT, t0=T0)

    # Simulation -----------------------------------------------------------------------------------------------------
    for i in range(int(np.floor(T_sim_total / DT))):
        # loop info
        if verbose:
            print('====================================================')
            print(f'simulation running at t = %.3f' % (i * DT + T0))

        # social force model
        f_total, fd, fv, ped_state = ped_sfm.transition(veh=veh_ego)
        if verbose:
            print(f'Social Force: state = {ped_state}, '
                  f'f_total = {fd.reshape(2)}(fd) + {fv.reshape(2)}(fv) = {f_total.reshape(2)}.')

        # pedestrian motion predictor
        prediction = predictor.predict(traj_past=ped.state.reshape(1, 4), t_pred=T_prediction, radius=ped.R)

        # vehicle control
        u, feasible = con_ego.generate_control(ref_speed=init_state[1], obj_pred=prediction)
        if verbose:
            print(f'Applied u = {u:.4f}.')

        # state update
        ped.update(f_total)
        veh_ego.update(u=u)

        # if collision, write collision info, terminate the loop
        if f_total[0][0] == float('Inf'):
            print('--------------------> Collision!')
            file_er = open(Path(result_path, 'collision_info.txt'), 'w')
            file_er.write(f'collision happened at t = {i * DT + T0:.3f}.')
            break

    # Save Data -----------------------------------------------------------------------------------------------------
    pickle_path = Path(result_path, f'sim_{control_method}_pos_{int(init_state[0]):3d}_vel_{int(init_state[1]):2d}.p')
    pickle.dump((layout, ped, veh_ego, ped_sfm, predictor, con_ego), open(pickle_path, "wb"))


if __name__ == '__main__':
    init_states = [-25, 8]
    control_method = 'mpc'  # select from 'mpc', 'oac', 'vkc'
    # control_method = 'oac'
    # control_method = 'vkc'
    simulate(init_state=init_states, control_method=control_method, verbose=True)


