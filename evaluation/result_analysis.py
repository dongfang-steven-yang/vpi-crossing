import numpy as np
import pickle
import matplotlib.pyplot as plt

control_method = 'pid'
predict_method = 'lin_last_obs'
date_time = '2020_01_28_21_29_23'
pos_range = [-40, -35, -30, -25, -20, -15]
vel_range = [2, 4, 6, 8, 10]
n_sim = 200

plt.ioff()

i_com = 0
for vel in vel_range:
    for pos in pos_range:
        i_com = i_com + 1
        print(f'-> ({control_method} in progress) combination ({i_com}/{len(vel_range) * len(pos_range)})')
        if np.abs((pos + 3.5) / vel) < 12:
            ped_gaps = np.empty(n_sim)
            ped_vds = np.empty(n_sim)
            veh_avg_vel = np.empty(n_sim)
            veh_max_abs_acc = np.empty(n_sim)
            veh_avg_abs_acc = np.empty(n_sim)
            init_state = [pos, vel]
            for num in range(n_sim):
                pickle_path = f'D:\\vpi_crossing_result\\{date_time}_sim_result_{control_method}_{predict_method}\\' \
                              f'pos {int(init_state[0]):3d} ' \
                              f'vel {int(init_state[1]):2d} ' \
                              f'itr {num:3d}.p'
                layout, ped, veh_ego, ped_sfm, predictor, con_ego = pickle.load(open(pickle_path, "rb"))
                # obtain data
                ped_gaps[num] = ped_sfm.thr_gap
                ped_vds[num] = ped_sfm.vd
                veh_avg_vel[num] = np.array(veh_ego.state_traj)[:, 3].mean()
                veh_max_abs_acc[num] = np.abs(np.array(veh_ego.action_traj)[:, 0]).max()
                veh_avg_abs_acc[num] = np.abs(np.array(veh_ego.action_traj)[:, 0]).mean()

            # pedestrian vd and gap sampling distribution
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.grid(linestyle='--', linewidth=0.5)
            ax.plot(ped_gaps, ped_vds, '.b', alpha=0.5, label='ped vd')
            ax.plot(ped_gaps, veh_max_abs_acc, '.r', alpha=0.5, label='veh_max_acc')
            ax.plot(ped_gaps, veh_avg_abs_acc, '.m', alpha=0.5, label='veh_avg_acc')
            ax.plot(ped_gaps, veh_avg_vel, '.g', alpha=0.5, label='veh_avg_vel')
            ax.set_title(f'{control_method} {predict_method} pos {int(init_state[0]):3d} vel {int(init_state[1]):2d}')
            ax.set_xlabel('gaps')
            ax.legend()
            fig.savefig(pickle_path[:-5] + '_gap_vd.png')
            plt.close(fig)
