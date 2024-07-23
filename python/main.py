import numpy as np
import matplotlib.pyplot as plt

from ekf import EKF

ekf = EKF(initial_x=0.1,initial_v=0.1,accel_variance=0.5)

plt.ion()
plt.figure()

DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_N_STEPS = 20

real_x = 0.0
real_v = 0.9
meas_variance = 0.1**2



mns = []
covs = []

for step in range(NUM_STEPS):
    covs.append(ekf.cov)
    mns.append(ekf.mean)
    
    real_x = real_x + DT*real_v
    
    ekf.predict(dt=DT)
    if step != 0 and step%MEAS_EVERY_N_STEPS == 0:
        ekf.update(meas_value=real_x + np.random.randn()*np.sqrt(meas_variance),
                   meas_variance=meas_variance)

    
plt.subplot(2, 1, 1)
plt.title('Position')
plt.plot([mn[0] for mn in mns], 'r')
plt.plot([mn[0]- 2*np.sqrt(cov[0,0]) for mn, cov in zip(mns, covs)], 'b--')
plt.plot([mn[0]+ 2*np.sqrt(cov[0,0]) for mn, cov in zip(mns, covs)], 'b--')

plt.subplot(2,1,2)
plt.title('Velocity')
plt.plot([mn[1] for mn in mns], 'r')
plt.plot([mn[1]+ 2*np.sqrt(cov[1,1]) for mn, cov in zip(mns, covs)], 'b--')
plt.plot([mn[1]- 2*np.sqrt(cov[1,1]) for mn, cov in zip(mns, covs)], 'b--')

# plt.subplot(2,1,2)
plt.show()
plt.ginput(1)