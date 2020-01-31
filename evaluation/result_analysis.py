import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

control_method = 'pid'
predict_method = 'lin_last_obs'
# date_time = 'pure_vk'
date_time = 'oac'

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
            # plot figure
            ped_gaps = np.empty(n_sim)
            ped_vds = np.empty(n_sim)
            veh_avg_vel = np.empty(n_sim)
            veh_max_abs_acc = np.empty(n_sim)
            veh_avg_abs_acc = np.empty(n_sim)
            mc = np.full(n_sim, False)  # mask of collision
            mg = np.full(n_sim, True)  # mask of positive gap
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
                if os.path.exists(collision_info_path):
                    mc[num] = True

                # trim gap
                if ped_sfm.thr_gap < 0:
                    mg[num] = False

                # record data
                ped_gaps[num] = ped_sfm.thr_gap
                ped_vds[num] = ped_sfm.vd
                veh_avg_vel[num] = np.array(veh_ego.state_traj)[:, 3].mean()
                veh_max_abs_acc[num] = np.abs(np.array(veh_ego.action_traj)[:, 0]).max()
                veh_avg_abs_acc[num] = np.abs(np.array(veh_ego.action_traj)[:, 0]).mean()

            # pedestrian vd and gap sampling distribution
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.grid(linestyle='--', linewidth=0.5)
            # ax.plot(ped_gaps[mg & ~mc], ped_vds[mg & ~mc], '.b', alpha=0.5, label='ped vd')
            if not date_time == 'pure_vk':
                ax.plot(ped_gaps[mg & ~mc], veh_max_abs_acc[mg & ~mc], '.r', alpha=0.5, label='veh_max_acc')
                ax.plot(ped_gaps[mg & ~mc], veh_avg_abs_acc[mg & ~mc], '.m', alpha=0.5, label='veh_avg_acc')
                ax.plot(ped_gaps[mg & ~mc], veh_avg_vel[mg & ~mc], '.g', alpha=0.5, label='veh_avg_vel')

            ax.plot(ped_gaps[mg & mc], veh_max_abs_acc[mg & mc], '^r', alpha=0.5)
            ax.plot(ped_gaps[mg & mc], veh_avg_abs_acc[mg & mc], '^m', alpha=0.5)
            ax.plot(ped_gaps[mg & mc], veh_avg_vel[mg & mc], '^g', alpha=0.5)

            ax.set_title(f'{control_method} {predict_method} pos {int(init_state[0]):3d} vel {int(init_state[1]):2d}')
            ax.set_xlabel('gaps')
            ax.legend()
            fig.savefig(pickle_path[:-5] + '_gap_vd.png')
            plt.close(fig)
