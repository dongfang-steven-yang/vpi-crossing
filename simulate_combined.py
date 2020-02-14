import numpy as np
from simulate import *
import datetime

pos_range = [-40, -35, -30, -25, -20, -15]
# pos_range = [-15]
vel_range = [2, 4, 6, 8, 10]
date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
control_methods = ['pid']
predict_method = 'lin_last_obs'

for control_method in control_methods:
    i_com = 0
    for vel in vel_range:
        for pos in pos_range:
            i_com = i_com + 1
            print(f'-> ({control_method} in progress) Enter combination ({i_com}/{len(vel_range) * len(pos_range)})')
            if np.abs((pos+3.5)/vel) < 12:
                for num in range(200):
                    init_state = [pos, vel]
                    print(f'--> Starting simulation with vehicle initial state = {init_state}, '
                          f'no. ({num}).')
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