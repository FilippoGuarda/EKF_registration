"""
The following function uses the cpp UKF class linked trough pybind
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import sys
ukf_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ukf_file_path + '/../cpp/build')

import ukf_cpy as ukf_cpp


ukf = ukf_cpp.UKF(3,3, np.ones((3,1))*0.1, np.eye(3)*100)

DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_N_STEPS = 2


real_x = np.array([1.0,1.0,np.pi/2])
real_velocity = - np.array([2.0,0.0, np.pi*0.01])
meas_variance = np.array([[1, 0.0, 0.0],
                          [0.0, 1, 0.0],
                          [0.0, 0.0, 0.03]])**2
measure = np.array([0.1, 0.2, 0.3])



mns = []
meas = []
covs = []
track = []

for step in range(NUM_STEPS):
    covs.append(ukf.cov)
    mns.append(ukf.mean)
    meas.append(measure)
    
    real_x = real_x + np.array([-real_velocity[0]*np.cos(real_x[2]) - real_velocity[1]*np.sin(real_x[2]),
                                -real_velocity[0]*np.sin(real_x[2]) - real_velocity[1]*np.cos(real_x[2]),
                                real_velocity[2]]) * DT
    noisy_velocity = np.array(real_velocity*np.random.randn(3)*0.1)
    ukf.predict(noisy_velocity, DT)
    if step != 0 and step%MEAS_EVERY_N_STEPS == 0:
        measure = real_x + (np.eye(3)@(np.random.randn(3)))@(np.sqrt(meas_variance.T))
        ukf.update(measure)
    else:
        ukf.update(measure)




    
mns = np.array(mns)
meas = np.array(meas)

plt.plot(meas[:, 0], meas[:,1], color='r', ls='--', lw=2)
plt.plot(mns[:, 0], mns[:,1], color='k', lw=1)
plt.axis('equal')
plt.title("ukf object position")

# plt.subplot(2,1,2)

plt.ginput(1)
