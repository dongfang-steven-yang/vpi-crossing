from matplotlib.animation import FuncAnimation, writers, FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Polygon
from pathlib import Path

# for arrow legends, check: https://stackoverflow.com/questions/22348229/matplotlib-legend-for-an-arrow
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches


class SimVisCrossing:

    def __init__(self, layout, ped, veh, sfm, predictor, con):
        # time step
        assert ped.dt == veh.dt == sfm.dt == predictor.dt == con.dt
        self.dt = ped.dt
        self.fps = 1 / self.dt

        # trajectory of time
        assert ped.t_traj[0] == veh.t_traj[0] == sfm.t_traj[0] == predictor.t_traj[0] == con.t_traj[0]
        assert ped.t == veh.t == sfm.t == predictor.t == con.t
        self.traj_t = ped.t_traj[:-1]

        # data
        self.layout = layout
        self.ped = ped
        self.veh = veh
        self.sfm = sfm
        self.pred = predictor
        self.con = con

        # transformed data for performance
        veh_state_by_time = np.array(self.veh.state_traj).swapaxes(0, 1)
        self.veh_state_x = veh_state_by_time[0]
        self.veh_state_vel = veh_state_by_time[3]
        self.veh_control_u = np.array(con.u_traj)
        self.ped_state = np.array(self.ped.state_traj).swapaxes(0, 1)
        self.gaps = np.array(self.sfm.gap_traj)
        self.gaps[self.gaps > 1000] = 1000

    def plot(self, save_figure=False, save_path='sim_test.png'):
        # config
        self.i_step = len(self.traj_t)
        if save_figure:
            plt.ioff()

        # figure scene
        self.fig_scene = plt.figure(figsize=(8, 5))
        ax1 = self.fig_scene.add_subplot(2, 2, 1)
        ax2 = self.fig_scene.add_subplot(2, 2, 2)
        ax3 = self.fig_scene.add_subplot(2, 2, 3)
        ax4 = self.fig_scene.add_subplot(2, 2, 4)
        self.fig_scene.subplots_adjust(left=0.08, bottom=0.15, right=0.98, top=0.98, wspace=0.19, hspace=0.29)

        frame_wait = np.argwhere(np.array(self.sfm.state_traj) == 'wait')[-1, 0]
        frames = [
            5, # approaching
            frame_wait,  # waiting
            frame_wait + 10,
            frame_wait + 20
        ]

        self._plot_main_scene_small(ax=ax1, step=frames[0])
        self._plot_main_scene_small(ax=ax2, step=frames[1])
        self._plot_main_scene_small(ax=ax3, step=frames[2])
        self._plot_main_scene_small(ax=ax4, step=frames[3], legend=True)

        # figure data
        self.fig_data = plt.figure(figsize=(8, 12))
        ax1s = self.fig_data.add_subplot(7, 1, 1)
        ax2s = self.fig_data.add_subplot(7, 1, 2)
        ax3s = self.fig_data.add_subplot(7, 1, 3)
        ax4s = self.fig_data.add_subplot(7, 1, 4)
        ax5s = self.fig_data.add_subplot(7, 1, 5)
        ax6s = self.fig_data.add_subplot(7, 1, 6)
        ax7s = self.fig_data.add_subplot(7, 1, 7)
        self.fig_data.subplots_adjust(left=0.1, bottom=0.04, right=0.99, top=0.98, wspace=0.13, hspace=0.84)

        # vehicle
        self._plot_veh_x_pos(ax1s)
        self._plot_veh_x_vel(ax2s)
        self._plot_veh_control(ax3s)

        # pedestrian
        self._plot_ped_y_pos(ax4s)
        self._plot_ped_speed(ax5s)
        self._plot_ped_state(ax6s)
        self._plot_ped_gap(ax7s)

        self.fig_scene.savefig(save_path[:-4] + '_scene.png', dpi=300)
        self.fig_scene.savefig(save_path[:-4] + '_scene.pdf')
        self.fig_data.savefig(save_path[:-4] + '_data.png', dpi=300)
        self.fig_data.savefig(save_path[:-4] + '_data.pdf')

        print('Result figure saved.')





    def animate(self, save_video=False, save_path='sim_test.mp4'):
        # figure
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.add_subplot(2, 1, 1)
        self.fig.add_subplot(4, 3, 7)
        self.fig.add_subplot(4, 3, 8)
        self.fig.add_subplot(4, 3, 9)
        self.fig.add_subplot(4, 3, 10)
        self.fig.add_subplot(4, 3, 11)
        self.fig.add_subplot(4, 3, 12)
        self.ax_gap = self.fig.axes[-1].twinx()
        self.fig.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.95, wspace=0.18, hspace=0.42)

        # state
        self.i_step = 0
        if save_video:
            plt.ioff()

        # animator
        ani = FuncAnimation(self.fig, self.update, frames=len(self.traj_t), interval=1000 / self.fps)

        if save_video:
            # Path(path_saving_video).mkdir(parents=True, exist_ok=True)
            # Set up formatting for the movie files
            Writer = writers['ffmpeg']
            writer = Writer(fps=1 / self.dt, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(save_path, writer=writer)
        else:
            plt.show()

    def update(self, t):
        # access axes
        ax_main = self.fig.axes[0]
        ax1 = self.fig.axes[1]
        ax2 = self.fig.axes[2]
        ax3 = self.fig.axes[3]
        ax4 = self.fig.axes[4]
        ax5 = self.fig.axes[5]
        ax6 = self.fig.axes[6]
        ax_gap = self.ax_gap

        # Main Scenario -------------------------------------------------------------
        self._plot_main_scene(ax=ax_main)

        # Vehicle Controller Information --------------------------------------------------------------
        self._plot_veh_x_pos(ax1)
        self._plot_veh_x_vel(ax2)
        self._plot_veh_control(ax3)

        # Pedestrian Information ----------------------------------------------------------------------
        self._plot_ped_y_pos(ax4)
        self._plot_ped_speed(ax5)
        self._plot_ped_state(ax6)
        self._plot_ped_gap_embedded(ax_gap)

        # next time step
        if self.i_step < len(self.traj_t) - 2:
            # print(self.i_step)
            self.i_step = self.i_step + 1
        else:
            print("All sequences have been displayed ! ")
            # self.fig.savefig(str(self.path_saving_last_frame + '.png'), dpi=300)
            # self.fig.savefig(str(self.path_saving_last_frame + '.pdf'))
            # sys.exit()

    def _get_veh_info(self, step=None):
        if step is None:
            step = self.i_step

        veh_C2R = self.veh.C2R
        veh_L = self.veh.LENGTH
        veh_W = self.veh.WIDTH
        x = self.veh.state_traj[step][0]
        y = self.veh.state_traj[step][1]
        yaw = self.veh.state_traj[step][2]
        vel = self.veh.state_traj[step][3]

        offsets = [[- veh_C2R, - veh_W / 2],
                   [- veh_C2R, veh_W / 2],
                   [veh_L - veh_C2R, veh_W / 2],
                   [veh_L - veh_C2R, - veh_W / 2]]
        offsets = np.array(offsets).swapaxes(0, 1)
        rotation_mat = np.array([[np.cos(yaw), - np.sin(yaw)],
                                 [np.sin(yaw), np.cos(yaw)]])
        offsets_converted = rotation_mat.dot(offsets)
        vertices = offsets_converted + np.array([[x], [y]])
        return vertices, x, y, yaw, vel

    def _plot_main_scene_small(self, ax, step, legend=False):
        # config
        x_min = -20
        x_max = 5
        y_min = -5
        y_max = 12

        ax.clear()
        # ax.title.set_text(f'Simulation Animation - time at {self.traj_t[step]:.3f} s')
        ax.set_xlabel('x position (m)')
        ax.set_ylabel('y position (m)')
        ax.axis('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(linestyle='--', linewidth=0.5)

        # layout
        for key, cts in self.layout.items():
            if key == 'lane_line_solid':
                for ct in cts:
                    ax.plot(ct[0], ct[1], '-g')
            elif key == 'lane_line_dashed':
                for ct in cts:
                    ax.plot(ct[0], ct[1], '--g')
            else:
                raise Exception('undifeind layout type.')

        # vehicle
            # --- shape
        vertices, x, y, yaw, vel = self._get_veh_info(step=step)
        polygon = Polygon(vertices.swapaxes(0, 1), fill=True, color='y')
        ax.add_artist(polygon)
        # --- position
        circle = plt.Circle((x, y), 0.2, color='#f08102')
        ax.add_artist(circle)
        # --- orientation
        ar_vel_veh = ax.arrow(
            x, y, vel * np.cos(yaw), vel * np.sin(yaw),
            width=0.05,
            # length_includes_head=True,
            linestyle='-', color='#f08102', head_width=0.3, head_length=0.5
        )

        # pedestrian prediction
        pred = self.pred.pred_traj[step]
        if pred is None:
            ax.text(-10, y_min + 1, f'no prediction result', color='r')
        else:
            # plot prediction
            # print(pred)
            for i, p in enumerate(pred):
                circle = plt.Circle((p[0], p[1]), p[4], color=(1.0, 0.5 + 0.5 * i / len(pred), 1.0))
                # circle = plt.Circle((p[0], p[1]), p[4], color='g')
                ax.add_artist(circle)

        # pedestrian
        state_ped = self.ped.state_traj[step]
        circle = plt.Circle((state_ped[0], state_ped[1]), self.ped.R, color='r')
        ax.add_artist(circle)
        alpha_arrow = 0.3
        ar_vel_ped = ax.arrow(
            state_ped[0], state_ped[1],
            state_ped[2], state_ped[3],
            # length_includes_head=True,
            width=0.03, color='k', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        )

        # pedestrian motion - social forces
        f_total = self.sfm.f_total_traj[step]
        fv = self.sfm.fv_traj[step]
        fd = self.sfm.fd_traj[step]
        state_sfm = self.sfm.state_traj[step]
        des = self.sfm.des_traj[step]
        gap = self.sfm.gap_traj[step]

        # scale = 20
        # ar1 = ax.arrow(
        #     state_ped[0], state_ped[1],
        #     (state_ped[0] + fv[0][0]) / scale, (state_ped[1] + fv[1][0]) / scale,
        #     # length_includes_head=True,
        #     width=0.03, color='c', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        # )
        # ar2 = ax.arrow(
        #     state_ped[0], state_ped[1],
        #     (state_ped[0] + fd[0][0]) / scale, (state_ped[1] + fd[1][0]) / scale,
        #     # length_includes_head=True,
        #     width=0.03, color='m', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        # )
        # ar3 = ax.arrow(
        #     state_ped[0], state_ped[1],
        #     (state_ped[0] + f_total[0][0]) / scale, (state_ped[1] + f_total[1][0]) / scale,
        #     # length_includes_head=True,
        #     width=0.03, color='b', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        # )

        ax.plot(des[0], des[1], 'xb', label='destination')

        # text
        ax.text(x_min, y_max - 2, '$t=%.2fs$' % self.traj_t[step])
        ax.text(x_min, y_max - 3.5, '$v_{veh}=%.2fMPH(%.2fm/s)$' % (vel * 2.237, vel))
        ax.text(x_min, y_max - 5, '$u_{veh}=%.2fm/s^2$' % self.con.u_traj[step])
        # ax.text(x_min, y_max - 5, '$f_{total}=[%.2f,%.2f]$' % (f_total[0][0],f_total[1][0]))
        # ax.text(x_min, y_max - 6.6, '$f_{veh}=[%.2f,%.2f]$' % (fv[0][0], fv[1][0]))
        # ax.text(x_min, y_max - 8.1, '$f_{des}=[%.2f,%.2f]$' % (fd[0][0], fd[1][0]))

        ax.text(x_min, -1.5, f'pedestrian status: {state_sfm}')
        ax.text(x_min, -3, '$\\tau_{gap}=%.2fs,t_{gap}=%.2fs,v_0=%.2fm/s$' % (self.sfm.thr_gap, gap, self.sfm.vd))


        # ax.text(x_min, -4.5, '$f_{total}=[%.2f,%.2f],f_{veh}=[%.2f,%.2f],f_{des}=[%.2f,%.2f]$'
        #         % (f_total[0][0],f_total[1][0],fv[0][0],fv[1][0],fd[0][0],fd[1][0]) )


        # legend
        if legend:
            ax.legend(
                # [ar_vel_veh, ar_vel_ped, ar1, ar2, ar3],
                [ar_vel_veh, ar_vel_ped],

                ['vehicle velocity', 'pedestrian velocity', 'f_vehicle', 'f_destination', 'f_total'],
                loc='upper center', bbox_to_anchor=(-0.15, -0.22), shadow=True, ncol=5,
                handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow), }
            )

            # plt.show()

    def _plot_main_scene(self, ax):
        step = self.i_step

        # config
        x_min = -50
        x_max = 10
        y_min = -5
        y_max = 10

        ax.clear()
        ax.title.set_text(f'Simulation Animation - time at {self.traj_t[step]:.3f} s')
        ax.set_xlabel('x position (m)')
        ax.set_ylabel('y position (m)')
        ax.axis('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(linestyle='--', linewidth=0.5)

        # layout
        for key, cts in self.layout.items():
            if key == 'lane_line_solid':
                for ct in cts:
                    ax.plot(ct[0], ct[1], '-g')
            elif key == 'lane_line_dashed':
                for ct in cts:
                    ax.plot(ct[0], ct[1], '--g')
            else:
                raise Exception('undifeind layout type.')

        # vehicle
        # --- shape
        vertices, x, y, yaw, vel = self._get_veh_info()
        polygon = Polygon(vertices.swapaxes(0, 1), fill=True, color='y')
        ax.add_artist(polygon)
        # --- position
        circle = plt.Circle((x, y), 0.2, color='#f08102')
        ax.add_artist(circle)
        # --- orientation
        ar_vel_veh = ax.arrow(
            x, y, vel * np.cos(yaw), vel * np.sin(yaw),
            width=0.05,
            # length_includes_head=True,
            linestyle='-', color='#f08102', head_width=0.3, head_length=0.5
        )

        # pedestrian prediction
        pred = self.pred.pred_traj[step]
        if pred is None:
            ax.text(-10, y_min + 1, f'no prediction result', color='r')
        else:
            # plot prediction
            # print(pred)
            for i, p in enumerate(pred):
                circle = plt.Circle((p[0], p[1]), p[4], color=(1.0, 0.5 + 0.5 * i / len(pred), 1.0))
                # circle = plt.Circle((p[0], p[1]), p[4], color='g')
                ax.add_artist(circle)

        # pedestrian
        state_ped = self.ped.state_traj[step]
        circle = plt.Circle((state_ped[0], state_ped[1]), self.ped.R, color='r')
        ax.add_artist(circle)
        alpha_arrow = 0.3
        ar_vel_ped = ax.arrow(
            state_ped[0], state_ped[1],
            state_ped[2], state_ped[3],
            # length_includes_head=True,
            width=0.03, color='k', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        )

        # pedestrian motion - social forces
        f_total = self.sfm.f_total_traj[step]
        fv = self.sfm.fv_traj[step]
        fd = self.sfm.fd_traj[step]
        state_sfm = self.sfm.state_traj[step]
        des = self.sfm.des_traj[step]
        gap = self.sfm.gap_traj[step]

        scale = 50
        ar1 = ax.arrow(
            state_ped[0], state_ped[1],
            (state_ped[0] + fv[0][0]) / scale, (state_ped[1] + fv[1][0]) / scale,
            # length_includes_head=True,
            width=0.03, color='c', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        )
        ar2 = ax.arrow(
            state_ped[0], state_ped[1],
            (state_ped[0] + fd[0][0]) / scale, (state_ped[1] + fd[1][0]) / scale,
            # length_includes_head=True,
            width=0.03, color='m', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        )
        ar3 = ax.arrow(
            state_ped[0], state_ped[1],
            (state_ped[0] + f_total[0][0]) / scale, (state_ped[1] + f_total[1][0]) / scale,
            # length_includes_head=True,
            width=0.03, color='b', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        )

        ax.plot(des[0], des[1], 'xb', label='destination')
        ax.legend([ar_vel_veh, ar_vel_ped, ar1, ar2, ar3],
                  ['vehicle velocity', 'pedestrian velocity', 'f_vehicle', 'f_destination', 'f_total'],
                  loc='upper center', bbox_to_anchor=(0.78, -0.08), shadow=True, ncol = 5,
                  handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow), })

        # text
        ax.text(x_min + 2, y_max - 1, f'vehicle speed = {vel * 2.237:.2f} MPH ({vel:.2f} m/s)')
        ax.text(x_min + 2, y_max - 2, f'vehicle control = {self.con.u_traj[step]:.2f} m/s^2')

        ax.text(-30, y_max - 1, f'pedestrian motion status: {state_sfm}')
        ax.text(-30, y_max - 2, f'gap acceptance = {self.sfm.thr_gap:.4f}; current gap = {gap:.4f} s; '
                                     f'pedestrian desired v = {self.sfm.vd:.4f} m/s')
        ax.text(-30, y_max - 3, f'f_total = [{f_total[0][0]:.4f},{f_total[1][0]:.4f}], '
                                     f'f_vehicle = [{fv[0][0]:.4f},{fv[1][0]:.4f}], '
                                     f'f_destination = [{fd[0][0]:.4f},{fd[1][0]:.4f}].')

    def _plot_veh_x_pos(self, ax):
        ax.clear()
        ax.title.set_text('Vehicle X Position')
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('position (m)')
        ax.grid(linestyle='--', linewidth=0.5)
        ax.plot(self.traj_t[:self.i_step], self.veh_state_x[:self.i_step], '-')
        ax.set_xlim(self.traj_t[0], self.traj_t[-1])
        ax.set_ylim(self.veh_state_x.min() - 5, self.veh_state_x.max() + 5)

    def _plot_veh_x_vel(self, ax):
        ax.clear()
        ax.title.set_text('Vehicle X Speed')
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('speed (m/s)')
        ax.grid(linestyle='--', linewidth=0.5)
        ax.plot(self.traj_t[:self.i_step], self.veh_state_vel[:self.i_step], '-')
        ax.set_xlim(self.traj_t[0], self.traj_t[-1])
        ax.set_ylim(self.veh_state_vel.min() - 5, self.veh_state_vel.max() + 5)

    def _plot_veh_control(self, ax):
        ax.clear()
        ax.title.set_text('Vehicle Control Action')
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('acceleration (m/s^2)')
        ax.grid(linestyle='--', linewidth=0.5)
        ax.plot(self.traj_t[:self.i_step], self.veh_control_u[:self.i_step], '-')
        ax.set_xlim(self.traj_t[0], self.traj_t[-1])
        ax.set_ylim(self.veh_control_u.min() - 5, self.veh_control_u.max() + 5)

    def _plot_ped_y_pos(self, ax):
        ax.clear()
        ax.title.set_text('Pedestrian Y Position')
        ax.grid(linestyle='--', linewidth=0.5)
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('position (m)')
        ax.plot(self.traj_t[:self.i_step], self.ped_state[1][:self.i_step], label='y position')
        ax.set_xlim(self.traj_t[0], self.traj_t[-1])
        ax.set_ylim(self.ped_state[1][:-1].min() - 1, self.ped_state[1][:-1].max() + 1)

    def _plot_ped_speed(self, ax):
        ax.clear()
        ax.title.set_text('Pedestrian Speed')
        ax.grid(linestyle='--', linewidth=0.5)
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('velocity (m/s)')
        # ax.plot(self.traj_t[:self.i_step], self.ped_state[3][:self.i_step], label='$|v_p|$')
        ax.plot(self.traj_t[:self.i_step], np.linalg.norm(self.ped_state[2:, :self.i_step], axis=0), label='$|v_p|$')
        ax.plot([self.traj_t[0], self.traj_t[-1]], [self.sfm.vd, self.sfm.vd], '--', label='$v_0$')
        ax.set_xlim(self.traj_t[0], self.traj_t[-1])
        ax.set_ylim(self.ped_state[3][:-1].min() - 0.5, self.ped_state[3][:-1].max() + 0.5)
        ax.legend()

    def _plot_ped_state(self, ax):
        ax.clear()
        ax.plot(self.traj_t[:self.i_step], self.sfm.state_traj[:self.i_step], '.')
        ax.title.set_text('Pedestrian State')
        ax.grid(linestyle='--', linewidth=0.5)
        ax.set_xlabel('time (sec)')
        ax.set_ylim(-1, 4)
        ax.set_xlim(self.traj_t[0], self.traj_t[-1])

    def _plot_ped_gap(self, ax):
        ax.clear()
        ax.grid(linestyle='--', linewidth=0.5)
        ax.set_ylabel('gap (sec)')
        ax.set_xlabel('time (sec)')
        ax.plot(self.traj_t[:self.i_step], self.sfm.gap_traj[:self.i_step], label='$t_{gap}$')
        ax.plot([self.traj_t[0], self.traj_t[-1]], [self.sfm.thr_gap, self.sfm.thr_gap], '--', label='$\\tau_{gap}$')
        ax.set_title('Pedestrian Accepted Gap')
        ax.set_ylim(0, 10)
        ax.set_xlim(self.traj_t[0], self.traj_t[-1])
        ax.legend()

    def _plot_ped_gap_embedded(self, ax):
        ax.clear()
        ax.grid(linestyle='--', linewidth=0.5)
        ax.plot(self.traj_t[:self.i_step], self.sfm.gap_traj[:self.i_step], '-r', label='$t_{gap}$')
        ax.plot([self.traj_t[0], self.traj_t[-1]], [self.sfm.thr_gap, self.sfm.thr_gap], '--', label='$\\tau_{gap}$')
        ax.set_ylim(0, 10)
        ax.set_ylabel('gap (sec)', color='r')
        ax.legend()


def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p