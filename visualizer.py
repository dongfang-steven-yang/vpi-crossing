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

    def animate(self, save_video=False, save_path='sim_test.mp4'):
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

        # config
        x_min = -50
        x_max = 10
        y_min = -5
        y_max = 10

        ax_main.clear()
        ax_main.title.set_text(f'Simulation Animation - time at {self.traj_t[self.i_step]:.3f} s')
        ax_main.set_xlabel('x position (m)')
        ax_main.set_ylabel('y position (m)')
        ax_main.axis('equal')
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)
        ax_main.grid(linestyle='--', linewidth=0.5)

        # layout
        for key, cts in self.layout.items():
            if key == 'lane_line_solid':
                for ct in cts:
                    ax_main.plot(ct[0], ct[1], '-g')
            elif key == 'lane_line_dashed':
                for ct in cts:
                    ax_main.plot(ct[0], ct[1], '--g')
            else:
                raise Exception('undifeind layout type.')

        # vehicle
        # --- shape
        vertices, x, y, yaw, vel = self._get_veh_info()
        polygon = Polygon(vertices.swapaxes(0, 1), fill=True, color='y')
        ax_main.add_artist(polygon)
        # --- position
        circle = plt.Circle((x, y), 0.2, color='#f08102')
        ax_main.add_artist(circle)
        # --- orientation
        ar_vel_veh = ax_main.arrow(
            x, y, vel * np.cos(yaw), vel * np.sin(yaw),
            width=0.05,
            # length_includes_head=True,
            linestyle='-', color='#f08102', head_width=0.3, head_length=0.5
        )
        # --- speed
        ax_main.text(x_min + 2, y_max - 1, f'vehicle speed = {vel * 2.237:.2f} MPH ({vel:.2f} m/s)')
        ax_main.text(x_min + 2, y_max - 2, f'vehicle control = {self.con.u_traj[self.i_step]:.2f} m/s^2')

        # pedestrian prediction
        pred = self.pred.pred_traj[self.i_step]
        if pred is None:
            ax_main.text(-10, y_min + 1, f'no prediction result', color='r')
        else:
            # plot prediction
            # print(pred)
            for i, p in enumerate(pred):
                circle = plt.Circle((p[0], p[1]), p[4], color=(1.0, 0.5 + 0.5 * i / len(pred), 1.0))
                # circle = plt.Circle((p[0], p[1]), p[4], color='g')
                ax_main.add_artist(circle)

        # pedestrian
        state_ped = self.ped.state_traj[self.i_step]
        circle = plt.Circle((state_ped[0], state_ped[1]), self.ped.R, color='r')
        ax_main.add_artist(circle)
        alpha_arrow = 0.3
        ar_vel_ped = ax_main.arrow(
            state_ped[0], state_ped[1],
            state_ped[2], state_ped[3],
            # length_includes_head=True,
            width=0.03, color='k', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        )

        # pedestrian motion - social forces
        f_total = self.sfm.f_total_traj[self.i_step]
        fv = self.sfm.fv_traj[self.i_step]
        fd = self.sfm.fd_traj[self.i_step]
        state_sfm = self.sfm.state_traj[self.i_step]
        des = self.sfm.des_traj[self.i_step]
        gap = self.sfm.gap_traj[self.i_step]

        ax_main.text(-30, y_max - 1, f'pedestrian motion status: {state_sfm}')
        ax_main.text(-30, y_max - 2, f'gap acceptance = {self.sfm.thr_gap:.4f}; current gap = {gap:.4f} s; '
                                     f'pedestrian desired v = {self.sfm.vd:.4f} m/s')
        ax_main.text(-30, y_max - 3, f'f_total = [{f_total[0][0]:.4f},{f_total[1][0]:.4f}], '
                                     f'f_vehicle = [{fv[0][0]:.4f},{fv[1][0]:.4f}], '
                                     f'f_destination = [{fd[0][0]:.4f},{fd[1][0]:.4f}].')

        scale = 50
        ar1 = ax_main.arrow(
            state_ped[0], state_ped[1],
            (state_ped[0] + fv[0][0]) / scale, (state_ped[1] + fv[1][0]) / scale,
            # length_includes_head=True,
            width=0.03, color='c', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        )
        ar2 = ax_main.arrow(
            state_ped[0], state_ped[1],
            (state_ped[0] + fd[0][0]) / scale, (state_ped[1] + fd[1][0]) / scale,
            # length_includes_head=True,
            width=0.03, color='m', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        )
        ar3 = ax_main.arrow(
            state_ped[0], state_ped[1],
            (state_ped[0] + f_total[0][0]) / scale, (state_ped[1] + f_total[1][0]) / scale,
            # length_includes_head=True,
            width=0.03, color='b', head_width=0.3, head_length=0.5, alpha=alpha_arrow
        )

        ax_main.plot(des[0], des[1], 'xb', label='destination')
        ax_main.legend([ar_vel_veh, ar_vel_ped, ar1, ar2, ar3],
                       ['vehicle velocity', 'pedestrian velocity', 'f_vehicle', 'f_destination', 'f_total'],
                       loc='upper center', bbox_to_anchor=(0.78, -0.08), shadow=True, ncol = 5,
                       handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow), })

        # Vehicle Controller Information --------------------------------------------------------------

        ax1.clear()
        ax1.title.set_text('Vehicle X Position')
        ax1.set_xlabel('time (sec)')
        ax1.set_ylabel('position (m)')
        ax1.grid(linestyle='--', linewidth=0.5)
        ax1.plot(self.traj_t[:self.i_step], self.veh_state_x[:self.i_step], '-')
        ax1.set_xlim(self.traj_t[0], self.traj_t[-1])
        ax1.set_ylim(self.veh_state_x.min() - 5, self.veh_state_x.max() + 5)

        ax2.clear()
        ax2.title.set_text('Vehicle X Speed')
        ax2.set_xlabel('time (sec)')
        ax2.set_ylabel('speed (m/s)')
        ax2.grid(linestyle='--', linewidth=0.5)
        ax2.plot(self.traj_t[:self.i_step], self.veh_state_vel[:self.i_step], '-')
        ax2.set_xlim(self.traj_t[0], self.traj_t[-1])
        ax2.set_ylim(self.veh_state_vel.min() - 5, self.veh_state_vel.max() + 5)

        ax3.clear()
        ax3.title.set_text('Vehicle Control Action')
        ax3.set_xlabel('time (sec)')
        ax3.set_ylabel('acceleration (m/s^2)')
        ax3.grid(linestyle='--', linewidth=0.5)
        ax3.plot(self.traj_t[:self.i_step], self.veh_control_u[:self.i_step], '-')
        ax3.set_xlim(self.traj_t[0], self.traj_t[-1])
        ax3.set_ylim(self.veh_control_u.min() - 5, self.veh_control_u.max() + 5)

        # Pedestrian Information ----------------------------------------------------------------------

        ax4.clear()
        ax4.title.set_text('Pedestrian Y Position')
        ax4.grid(linestyle='--', linewidth=0.5)
        ax4.set_xlabel('time (sec)')
        ax4.set_ylabel('position (m)')
        ax4.plot(self.traj_t[:self.i_step], self.ped_state[1][:self.i_step], label='y position')
        ax4.set_xlim(self.traj_t[0], self.traj_t[-1])
        ax4.set_ylim(self.ped_state[1][:-1].min() - 1, self.ped_state[1][:-1].max() + 1)

        ax5.clear()
        ax5.title.set_text('Pedestrian Y Velocity')
        ax5.grid(linestyle='--', linewidth=0.5)
        ax5.set_xlabel('time (sec)')
        ax5.set_ylabel('velocity (m/s)')
        ax5.plot(self.traj_t[:self.i_step], self.ped_state[3][:self.i_step], label='y velocity')
        ax5.set_xlim(self.traj_t[0], self.traj_t[-1])
        ax5.set_ylim(self.ped_state[3][:-1].min() - 0.5, self.ped_state[3][:-1].max() + 0.5)

        ax6.clear()
        ax6.plot(self.traj_t[:self.i_step], self.sfm.state_traj[:self.i_step], '.')
        ax6.title.set_text('Pedestrian State')
        ax6.grid(linestyle='--', linewidth=0.5)
        ax6.set_xlabel('time (sec)')
        ax6.set_ylim(-1, 4)
        ax6.set_xlim(self.traj_t[0], self.traj_t[-1])

        ax_gap.clear()
        ax_gap.grid(linestyle='--', linewidth=0.5)
        ax_gap.plot(self.traj_t[:self.i_step], self.sfm.gap_traj[:self.i_step], '-r')
        ax_gap.set_ylim(0, 10)
        ax_gap.set_ylabel('gap (sec)', color='r')


        # next time step
        if self.i_step < len(self.traj_t) - 2:
            # print(self.i_step)
            self.i_step = self.i_step + 1
        else:
            print("All sequences have been displayed ! ")
            # self.fig.savefig(str(self.path_saving_last_frame + '.png'), dpi=300)
            # self.fig.savefig(str(self.path_saving_last_frame + '.pdf'))
            # sys.exit()

    def _get_veh_info(self):

        veh_C2R = self.veh.C2R
        veh_L = self.veh.LENGTH
        veh_W = self.veh.WIDTH
        x = self.veh.state_traj[self.i_step][0]
        y = self.veh.state_traj[self.i_step][1]
        yaw = self.veh.state_traj[self.i_step][2]
        vel = self.veh.state_traj[self.i_step][3]

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


def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p