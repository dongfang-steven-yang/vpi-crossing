import pickle
from tools.visualization import SimVisCrossing

data_path = 'results/sim_mpc_pos_-25_vel_ 8.p'


def evaluate(data_path):
    output_path = data_path
    # read data
    layout, ped, veh_ego, ped_sfm, predictor, con_ego = pickle.load(open(data_path, "rb"))
    vis = SimVisCrossing(layout=layout, ped=ped, veh=veh_ego, sfm=ped_sfm, predictor=predictor, con=con_ego)
    vis.animate(save_video=True, save_path=str(output_path)[:-1] + 'mp4')
    print('Video generated.')
    vis.plot(save_figure=True, save_path=str(output_path)[:-1] + 'png')
    print('Figures saved.')


if __name__ == '__main__':
    evaluate(data_path=data_path)
