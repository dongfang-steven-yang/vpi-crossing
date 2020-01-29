from simulate import *


if __name__ == '__main__':
    nums = [33, 41, 93, 105, 106, 158, 167, 176, 189]
    for num in nums:
        print(f'playback on num = {num}')
        playback(
            init_state=[-15, 10],
            control_method='pid',
            predict_method='lin_last_obs',
            num=num,
            date_time='2020_01_28_21_29_23'
        )
