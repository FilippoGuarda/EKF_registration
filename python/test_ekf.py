import numpy as np
from ekf import EKF
from unittest import TestCase



class TestKF(TestCase):
    
    def can_construct_xv(self):
        x = 0.2
        v = 0.5
        
        ekf = EKF(initial_x=x, initial_v=v, accel_variance=1.2)
        self.assertAlmostEqual(ekf.x[0], x)
        self.assertAlmostEqual(ekf.x[1], v)
        
    def can_predict_mean_P(self):
        
        x=0.2
        v=0.5
        
        ekf = EKF(initial_x=x, initial_v=v, accel_variance=1.2)
        ekf.predict(dt=0.1)
        
        self.assertEqual(ekf.cov.shape, (2,2))
        self.assertEqual(ekf.mean.shape, (2,))
        
    def can_update(self):
        
        x = 0.2
        v = 0.5
        
        ekf = EKF(initial_x=x, initial_v=v, accel_variance=1.2)
        ekf.update(meas_value=0.1, meas_variance=0.1)
        
        