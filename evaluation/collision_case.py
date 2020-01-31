from simulate import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

control_method = 'pid'
predict_method = 'lin_last_obs'
date_time = 'oac'
pos_range = [-40, -35, -30, -25, -20, -15]
vel_range = [2, 4, 6, 8, 10]
n_sim = 200
plt.ioff()

def save_collision_videos():
    i_com = 0
    for vel in vel_range:
        for pos in pos_range:
            i_com = i_com + 1
            print(f'-> ({control_method} in progress) combination ({i_com}/{len(vel_range) * len(pos_range)})')
            if np.abs((pos + 3.5) / vel) < 12:

                init_state = [pos, vel]

                i_plot = 5
                for num in range(n_sim):
                    # read data
                    pickle_path = f'D:\\vpi_crossing_result\\{date_time}_sim_result_{control_method}_{predict_method}\\' \
                                  f'pos {int(init_state[0]):3d} ' \
                                  f'vel {int(init_state[1]):2d} ' \
                                  f'itr {num:3d}.p'
                    layout, ped, veh_ego, ped_sfm, predictor, con_ego = pickle.load(open(pickle_path, "rb"))
                    # collision info
                    collision_info_path = pickle_path[:-2] + '_collision_info.txt'
                    if os.path.exists(collision_info_path) and \
                            ped_sfm.thr_gap > 0 and init_state == [-15, 6]:
                        print(f'playback on num = {num}')
                        playback(
                            init_state=init_state,
                            control_method=control_method,
                            predict_method=predict_method,
                            num=num,
                            date_time=date_time
                        )
                        i_plot = i_plot - 1


def anaylze_particular_one(pos, vel, num):
    init_state = [pos, vel]

    for num in range(n_sim):
        # read data
        pickle_path = f'D:\\vpi_crossing_result\\{date_time}_sim_result_{control_method}_{predict_method}\\' \
                      f'pos {int(init_state[0]):3d} ' \
                      f'vel {int(init_state[1]):2d} ' \
                      f'itr {num:3d}.p'
        layout, ped, veh_ego, ped_sfm, predictor, con_ego = pickle.load(open(pickle_path, "rb"))
        # collision info
        collision_info_path = pickle_path[:-2] + '_collision_info.txt'

        #
        t_traj = con_ego.t_traj
        u_traj = con_ego.u_traj
        feasible = con_ego.feasible_traj

        for i in range(len(t_traj)):
            print(f'At time t={t_traj[i]:.3f}, u={u_traj[i]:.3f}, feasible={feasible[i]}')

        # print(f'playback on num = {num}')
        # playback(
        #     init_state=init_state,
        #     control_method='mpc',
        #     predict_method='lin_last_obs',
        #     num=num,
        #     date_time='001'
        # )


if __name__ == '__main__':
    save_collision_videos()
    # anaylze_particular_one(pos=-15, vel=6, num=62)

