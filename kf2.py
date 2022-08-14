import numpy as np
import matplotlib.pyplot as plt

class KF(object):
    def __init__(self, dt, u_x,u_y, accel_variance, x_from_data, y_from_data):

        self.dt = dt

        #U matrix for storing velocities
        self.u = np.matrix([[u_x],[u_y]])

        # Intial State matrix
        self.x = np.matrix([[0], [0], [0], [0]])
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Defining Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        #Q is Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * accel_variance**2

        #R is Measurement Noise Covariance
        self.R = np.matrix([[x_from_data**2,0],
                           [0, y_from_data**2]])

        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])
    def predict(self):   
        # predicting new state
        #x_k =Ax_(k-1) + Bu_(k-1)  
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q           
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z):
        # S = H*P*Ht+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  

        new_x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))  
        I = np.eye(self.H.shape[1])
        # Update error covariance matrix
        new_P = (I - (K * self.H)) * self.P  
        self.x=new_x
        self.P=new_P
        

