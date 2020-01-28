import numpy as np
import matplotlib.pyplot as plt

t = [-9.9, -5.8, -5.6, -5.2, -4.2, -3.9, -3.8, -2.7, -2.6, -2.5, -1.5, -1.2, -0.2, -0.02, 0.7]
veh_speed = [35, 44, 44, 45, 45, 45, 45, 45, 45, 45, 44, 43, 40, 39, 37]
classification = [None, None, 1, ]


fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.grid(linestyle='--', linewidth=0.5)
ax.plot(t, veh_speed, '.--', label=f'vehicle speed')
ax.legend()
ax.set_ylim(0, 50)
ax.title.set_text('Accident')
ax.set_xlabel('time (sec) relative to impact')
ax.set_ylabel('speed (mph)')
plt.show()

print('testing')
