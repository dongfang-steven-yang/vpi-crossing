import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from pathlib import Path

control_method = ['vkc', 'oac', 'mpc']
predict_method = 'lin_last_obs'
# date_time = 'pure_vk'
date_time = 'now'

pos_range = [-20, -40, -35, -30, -25, -15]
vel_range = [10, 8, 6, 4, 2]
n_sim = 200
alpha = 0.3
size_tri = 20
plt.ioff()


dict_control = {
    'mpc': 'MPC',
    'oac': 'OAC',
    'vkc': 'VKC'
}


def get_min_dist(veh, ped, collision):
    if collision:
        return -1
    else:
        c2f = veh.C2F
        c2r = veh.C2R
        R = ped.R
        hw = veh.WIDTH / 2

        md = float('Inf')
        for i in range(len(veh.state_traj)):
            x_p = ped.state_traj[i][0]
            y_p = ped.state_traj[i][1]
            x_v = veh.state_traj[i][0]
            y_v = veh.state_traj[i][1]
            front = x_v + c2f + R
            rear = x_v - c2r - R
            left = y_v + hw + R
            right = y_v - hw - R
            if x_p > front:
                if y_p > left:
                    d = np.linalg.norm([x_p - front, y_p - left])
                elif y_p < right:
                    d = np.linalg.norm([x_p - front, y_p - right])
                else:
                    d = x_p - front
            elif front > x_p > rear:
                if y_p > left:
                    d = y_p - left
                elif y_p < right:
                    d = right - y_p
                else:
                    d = -1
            else:  # x < rear
                if y_p > left:
                    d = np.linalg.norm([x_p - rear, y_p - left])
                elif y_p < right:
                    d = np.linalg.norm([x_p - rear, y_p - right])
                else:
                    d = rear - x_p
            assert d >= 0
            if d < md:
                md = d
        return md


def result_analysis():
    i_com = 0
    for pos in pos_range:
        fig = plt.figure(figsize=(15, 10))
        fig.subplots_adjust(left=0.04, bottom=0.05, right=0.99, top=0.97, wspace=0.27, hspace=0.31)
        for i, vel in enumerate(vel_range):
            # print info
            i_com = i_com + 1
            print(f'-> (in progress) combination ({i_com}/{len(vel_range) * len(pos_range)})')
            if np.abs((pos + 3.5) / vel) < 12:
                init_state = [pos, vel]
                for ctrl in control_method:

                    # initialize data variables
                    ped_gaps = np.empty(n_sim)
                    ped_vds = np.empty(n_sim)
                    dists_min = np.empty(n_sim)
                    veh_avg_vel = np.empty(n_sim)
                    veh_max_abs_acc = np.empty(n_sim)
                    veh_avg_abs_acc = np.empty(n_sim)
                    mc = np.full(n_sim, False)  # mask of collision
                    mg = np.full(n_sim, True)  # mask of positive gap

                    # loop iter
                    for num in range(n_sim):
                        # read data
                        pickle_path = f'D:\\vpi_crossing_result\\{date_time}_sim_result_{ctrl}_{predict_method}\\' \
                                      f'pos {int(init_state[0]):3d} ' \
                                      f'vel {int(init_state[1]):2d} ' \
                                      f'itr {num:3d}.p'
                        layout, ped, veh_ego, ped_sfm, predictor, con_ego = pickle.load(open(pickle_path, "rb"))
                        # collision info
                        collision_info_path = pickle_path[:-2] + '_collision_info.txt'
                        if os.path.exists(collision_info_path):
                            mc[num] = True
                            collision = True
                        else:
                            collision = False

                        # trim gap
                        if ped_sfm.thr_gap < 0:
                            mg[num] = False

                        # record data
                        dists_min[num] = get_min_dist(veh_ego, ped, collision)
                        ped_gaps[num] = ped_sfm.thr_gap
                        ped_vds[num] = ped_sfm.vd
                        veh_avg_vel[num] = np.array(veh_ego.state_traj)[:, 3].mean()
                        veh_max_abs_acc[num] = np.abs(np.array(veh_ego.action_traj)[:, 0]).max()
                        veh_avg_abs_acc[num] = np.abs(np.array(veh_ego.action_traj)[:, 0]).mean()

                    # safety - collision / closest dist
                    ax = fig.add_subplot(3, 5, 0 + i + 1)
                    ax.grid(linestyle='--', linewidth=0.5)
                    ax.plot(ped_gaps[mg & ~mc], dists_min[mg & ~mc], '.', alpha=alpha, label=dict_control[ctrl])
                    ax.plot(ped_gaps[mg & mc], dists_min[mg & mc], '^', alpha=alpha, label=dict_control[ctrl]+'-C')
                    ax.set_title('$d_{front,0}=%.1f$, $\\dot{s}_0=%d$' % (abs(pos) - 3.5, vel))
                    ax.set_ylim(-1.5, 8)
                    ax.set_xlim(-0.5, 12.5)
                    ax.set_xlabel('Thr. of Accepted Time Gap (sec)')
                    ax.set_ylabel('Minimum Distance (m)')
                    ax.legend()

                    # efficiency - average speed
                    ax = fig.add_subplot(3, 5, 5 + i + 1)
                    ax.grid(linestyle='--', linewidth=0.5)
                    ax.plot(ped_gaps[mg & ~mc], veh_avg_vel[mg & ~mc], '.', alpha=alpha, label=dict_control[ctrl])
                    ax.plot(ped_gaps[mg & mc], veh_avg_vel[mg & mc], '^', alpha=alpha, label=dict_control[ctrl]+'-C')
                    ax.set_title('$d_{front,0}=%.1f$, $\\dot{s}_0=%d$' % (abs(pos) - 3.5, vel))
                    ax.set_ylim(0, 11)
                    ax.set_xlim(-0.5, 12.5)
                    ax.set_xlabel('Thr. of Accepted Time Gap (sec)')
                    ax.set_ylabel('Average Velocity ($m/s$)')
                    ax.legend()

                    # smoothness - maximum acceleration
                    ax = fig.add_subplot(3, 5, 10 + i + 1)
                    ax.grid(linestyle='--', linewidth=0.5)
                    ax.plot(ped_gaps[mg & ~mc], veh_max_abs_acc[mg & ~mc], '.', alpha=alpha, label=dict_control[ctrl])
                    ax.plot(ped_gaps[mg & mc], veh_max_abs_acc[mg & mc], '^', alpha=alpha, label=dict_control[ctrl]+'-C')
                    ax.set_title('$d_{front,0}=%.1f$, $\\dot{s}_0=%d$' % (abs(pos) - 3.5, vel))
                    ax.set_ylim(0, 7.5)
                    ax.set_xlim(-0.5, 12.5)
                    ax.set_xlabel('Thr. of Accepted Time Gap (sec)')
                    ax.set_ylabel('Maximum Acceleration ($m/s^2$)')
                    ax.legend()

        fig.savefig(str(Path(Path(pickle_path).parent.parent, str(pos) + '.png')), dpi=300)
        fig.savefig(str(Path(Path(pickle_path).parent.parent, str(pos) + '.pdf')))
        plt.close(fig)


if __name__ == '__main__':
    result_analysis()