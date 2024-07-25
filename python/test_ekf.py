import numpy as np
from ekf import EKF
from unittest import TestCase



class TestKF(TestCase):
    
    def can_construct_xv(self):
        x = np.array([0.1, 0.3, 0.4])
        
        ekf = EKF(initial_x=x, velocity_variance= np.array(0.1,0.1,0.1))
        self.assertAlmostEqual(ekf.x[0], x[0])
        self.assertAlmostEqual(ekf.x[1], x[1])
        self.assertAlmostEqual(ekf.x[2], x[2])
        
    def can_predict_mean_P(self):
        
        x = np.array([0.1, 0.3, 0.4])
        v = np.array([1, 4, 1])
        
        ekf = EKF(initial_x=x, velocity_variance= np.array(0.1,0.1,0.1))
        ekf.predict(dt=0.1)
        
        self.assertEqual(ekf.cov.shape, (3,3))
        self.assertEqual(ekf.mean.shape, (3,))
        
    def can_update(self):
        
        x = np.array([0.1, 0.3, 0.4])
        
        ekf = EKF(initial_x=x, velocity_variance=1.2)
        ekf.update(meas_value=np.array([0.1, 0.3, 0.6]), meas_variance=np.array([0.1, 0.1, 0.1]))       
        