import numpy as np
import matplotlib.pyplot as plt

from ekf import EKF

ekf = EKF(initial_x=np.array([0.0,0.0,0.0]),velocity_variance=np.array([0.1, 0.1, 0.1]))

DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_N_STEPS = 10


real_x = np.array([1.0,1.0,np.pi/2])
real_velocity = - np.array([2.0,0.0, np.pi*0.01])
meas_variance = np.array([[1, 0.0, 0.0],
                          [0.0, 1, 0.0],
                          [0.0, 0.0, 0.3]])**2
measure = np.array([0.1, 0.2, 0.3])



mns = []
meas = []
covs = []
track = []

for step in range(NUM_STEPS):
    covs.append(ekf.cov)
    mns.append(ekf.mean)
    meas.append(measure)
    
    real_x = real_x + np.array([-real_velocity[0]*np.cos(real_x[2]) - real_velocity[1]*np.sin(real_x[2]),
                                -real_velocity[0]*np.sin(real_x[2]) - real_velocity[1]*np.cos(real_x[2]),
                                -real_velocity[2]]) * DT
    
    ekf.predict(velocity=real_velocity*np.random.randn(3)*0.1, dt=DT)
    if step != 0 and step%MEAS_EVERY_N_STEPS == 0:
        measure = real_x + (np.eye(3)@(np.random.randn(3)))@(np.sqrt(meas_variance.T))
        ekf.update(meas_value= measure,
                   meas_variance=meas_variance)
    else:
        ekf.update(meas_value= measure,
                   meas_variance=meas_variance)



    
mns = np.array(mns)
meas = np.array(meas)

plt.plot(meas[:, 0], meas[:,1], color='r', ls='--', lw=1)
plt.plot(mns[:, 0], mns[:,1], color='k', lw=2)
plt.axis('equal')
plt.title("EKF object position")

# plt.subplot(2,1,2)

plt.ginput(1)