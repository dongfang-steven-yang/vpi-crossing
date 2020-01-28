import numpy as np
from simulate import *
import datetime

pos_range = list(np.arange(-40, -14, 5))
vel_range = list(np.arange(2, 10.1, 2))
date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

veh_init_state = [-40, 20 / R_MPH2mps]  # todo @ add more combinations

control_methods = ['mpc', 'pid']
predict_method = 'lin_last_obs'

for control_method in control_methods:
    i_com = 0
    for vel in vel_range:
        for pos in pos_range:
            i_com = i_com + 1
            print(f'-> ({control_method} in progress) Enter combination ({i_com}/{len(vel_range) * len(pos_range)})')
            if np.abs(pos/vel) < 12:
                for num in range(200):
                    init_state = [pos, vel]
                    print(f'--> Starting simulation with vehicle initial state = {init_state}, '
                          f'repeat ({num+1}/200).')
                    sim_once(
                        init_state=init_state,
                        control_method=control_method,
                        predict_method=predict_method,
                        num=num,
                        date_time=date_time,
                        suppress_video_save=True
                    )
            else:
                print('initial gap is larger than 12 seconds, skipped')