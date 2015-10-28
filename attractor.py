import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Attractor(object):
    def __init__(self, s=10, b=(8/3), p=28, start=0.0, end=80.0, points=10000):
        self.s = s
        self.b = b
        self.p = p
        self.start = start
        self.end = end
        self.points = points
        self.params = np.array(self.s, self.b, self.p)
        self.dt = (self.end - self.start)/self.points
        
    def euler(self, a):
        a = np.array([x,y,z])
        dx1 = self.s*(y-x)*self.dt
        dy1 = (x*(self.p-z)-y)*self.dt
        dz1 = (x*y-self.b*z)*self.dt
        return np.array([dx1,dy1,dz1])
    
    def rk2(self, a):
        a = np.array([x,y,z])
        new_inc = self.dt/2
        k1 = self.euler(a)
        
        dx2 = (self.s*(y-x)*(self.dt+new_inc)+(k1*new_inc)
        dy2 = (x*(self.p-z)-y)*(self.dt+new_inc)+(k1*new_inc)
        dz2 = (x*y-self.b*z)*(self.d+new_inc)+(k1*new_inc)
        return np.array([dx2,dy2,dz2])
              
    def rk3(self, a):
        a = np.array([x,y,z])
        new_inc = self.dt/2
        k2 = self.rk2(a)
        
        dx3 = (self.s*(y-x)*(self.dt+new_inc)+(k2*new_inc)
        dy3 = (x*(self.p-z)-y)*(self.dt+new_inc)+(k2*new_inc)
        dz3 = (x*y-self.b*z)*(self.d+new_inc)+(k2*new_inc)
        return np.array([dx3,dy3,dz3])
    
    def rk4(self, a):
        a = np.array([x,y,z])
        new_inc = self.dt
        k3 = self.rk3(a)
        
        dx4 = (self.s*(y-x)*(self.dt+new_inc)+(k3*new_inc)
        dy4 = (x*(self.p-z)-y)*(self.dt+new_inc)+(k3*new_inc)
        dz4 = (x*y-self.b*z)*(self.d+new_inc)+(k3*new_inc)
        return np.array([dx4,dy4,dz4])
    
    def evolve(self, r0 = np.array([0.1,0.0,0.0]), order = 4):
        #
        #
        #
        #result = 
        df = pd.DataFrame(result)
        df.columns = ["t","x","y","z"]
        self.solution = df
        return df
               
    def save(self):
        self.solution.to_csv("out.csv")
               
    def plotxy(self):
        plt.plot(self.solution["t"], self.solution["x"], "r")
        plt.plot(self.solution["t"], self.solution["y"], "b")
        plt.show()    
               
    def plotyz(self):
        plt.plot(self.solution["t"], self.solution["y"], "r")
        plt.plot(self.solution["t"], self.solution["z"], "b")
        plt.show()       

    def plotzx(self):       
        plt.plot(self.solution["t"], self.solution["z"], "r")
        plt.plot(self.solution["t"], self.solution["x"], "b")
        plt.show()    
    
    def plot3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.show()