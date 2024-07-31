import numpy as np
from unittest import TestCase

class EKF:
    
    def __init__(self, initial_x:np.array,
                        velocity_variance:np.array) -> None:
        
        # mean of state GRV
        self.x = initial_x
        # cov of state GRV
        self.velocity_variance = velocity_variance
        
        self.P = np.eye(3)        
        
    def predict(self, velocity: np.array, dt: float) -> None:
        # x = F * x 
        # P = F P Ft  
        F = np.array([[1, 0, velocity[0]*np.sin(self.x[2])-velocity[1]*np.cos(self.x[2])],
                      [0, 1, -velocity[0]*np.cos(self.x[2])+velocity[1]*np.sin(self.x[2])],
                      [0, 0, 1]])
        
        V = np.array([[-np.cos(self.x[2]), -np.sin(self.x[2]), velocity[0]*np.sin(self.x[2])-velocity[1]*np.cos(self.x[2])],
                    [-np.sin(self.x[2]), -np.cos(self.x[2]), -velocity[0]*np.cos(self.x[2])+velocity[1]*np.sin(self.x[2])],
                    [0,0,-1]])
        
        M = np.eye(3)*self.velocity_variance
        
        new_x = np.array([-velocity[0]*np.cos(self.x[2]) - velocity[1]*np.sin(self.x[2]),
                         -velocity[0]*np.sin(self.x[2]) - velocity[1]*np.cos(self.x[2]),
                         -velocity[2]])*dt
        
        
        new_P = F@self.P@F.T + V@M@V.T
        
        self.P = new_P
        self.x = self.x + new_x
        
    def residual(self, a, b):
        """ 
        This function takes care of pi, so 0 - pi/2 = 3/4 pi
        """
        y = a - b
        y[2] = y[2] % (2 * np.pi)    # force in range [0, 2 pi)
        if y[2] > np.pi:             # move to [-pi, pi)
            y[2] -= 2 * np.pi
        return y
        
    def update(self, meas_value:np.array, meas_variance:np.array) -> None:
        # y = z - H x
        # S = HPHt + R
        # K = PHt S-1
        # x = x + Ky
        # P = (I - KH)P
        
        H = np.eye(3)
        
        z = meas_value
        R = meas_variance
        y=np.zeros(3)
        
        y = self.residual(z, H@self.x)
        S = H@self.P@H.T + R
        K = self.P@H.T@(np.linalg.inv(S))
        
        new_x = self.x + K@y
        new_P = (np.eye(3) - K@H)@self.P
        
        self.P = new_P
        self.x = new_x
        
        
        
    @property
    def cov(self) -> np.array: 
        return self.P   
        
    @property
    def mean(self) -> np.array: 
        return self.x  
    
    @property
    def pos_x(self):
        return self.x[0]
    
    @property
    def pos_y(self):
        return self.x[1]
    
    @property
    def theta(self):
        return self.x[2]
    
    @property
    def vel(self):
        return self.x[1]