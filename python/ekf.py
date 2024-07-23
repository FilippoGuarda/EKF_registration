import numpy as np
from unittest import TestCase

class EKF:
    
    def __init__(self, initial_x:float,
                        initial_v:float,
                        accel_variance:float) -> None:
        
        # mean of state GRV
        self.x = np.array([initial_x, initial_v])

        # cov of state GRV
        self.accel_variance = accel_variance
        
        self.P = np.eye(2)        
        
    def predict(self, dt: float) -> None:
        # x = F * x 
        # P = F P Ft  
        F = np.array([[1, dt], [0, 1]])
        new_x = F.dot(self.x)
        
        G = np.array([0.5*dt**2, dt]).reshape((1,2))
        new_P = F.dot(self.P).dot(F.T) + G.dot(G.T) * self.accel_variance
        
        self.P = new_P
        self.x = new_x
        
    def update(self, meas_value:float, meas_variance:float) -> None:
        # y = z - H x
        # S = HPHt + R
        # K = PHt S-1
        # x = x + Ky
        # P = (I - KH)P
        
        H = np.array([1, 0]).reshape((1,2))
        
        z = np.array([meas_value])
        R = np.array([meas_variance])
        
        y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        
        new_x = self.x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self.P)
        
        self.P = new_P
        self.x = new_x
        
        
        
    @property
    def cov(self) -> np.array: 
        return self.P   
        
    @property
    def mean(self) -> np.array: 
        return self.x  
    
    @property
    def pos(self):
        return self.x[0]
    
    @property
    def vel(self):
        return self.x[1]