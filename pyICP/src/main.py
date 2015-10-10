#! /usr/bin/python
'''
Created on 14/08/2014

@author: Francisco Dominguez
Easy ICP a exercise in 2D in order to learn the basic concepts
Iterative Closest Point Algorithm
In this exercise you will use a standard ICP algorithm with the point-to-point distance metric to
estimate the transform between the 2D datasets (model - red and target - green) depicted in
the below figure. For the correspondence estimation please use the nearest neighbor search
with the maximum radius set to 4 grid units. For the rotation estimation use an SVD library of
your choice or calculate it yourself. Recall the main steps in the ICP algorithm:
- find point pairs;
- compute centroids
- build the correlation matrix H;
- estimate rotation matrix R using SVD;
- estimate the translation vector t;
- calculate the transform T;
- transform dataset m using the transform T.
'''
import numpy as np
import myICP as icp
import math
from visual import points,curve,rate,sphere,box
try:
    import cPickle as pickle
except ImportError:
    import pickle

for i in range(400):
    curve(pos=[(-2000,(i-400)*10),(2000,(i-400)*10)],color=(0.3,0.3,0.3))
for j in range(400):
    curve(pos=[((j-400)*10,-2000),((j-400)*10,2000)],color=(0.3,0.3,0.3))

#extract x,y,0 from Neato xv11 lidar in raw format 
def getModel(l):
    m=[]
    for angle,data in enumerate(l):
        if not data[2]:
            dist_mm = data[0] # distance is coded on 13 bits ? 14 bits ?
            quality = data[1] # quality is on 16 bits
            angle_rad = angle * math.pi / 180.0
            c =  math.cos(angle_rad)
            s = -math.sin(angle_rad)
            dist_x = dist_mm*c
            dist_y = dist_mm*s
            #box(pos=(dist_x,50,dist_y),width=150,height=100,length=150,color=(1,0,0))
            m.append((dist_x,dist_y,0))
    return np.array(m)
def getPickleModel(sf):
    f=file(sf)
    lidarJ0=pickle.load(f)
    f.close()
    return getModel(lidarJ0)
     
#dataset to be fitted in the target usually is a 2d range points or a 3D scanner point cloud    

#Madrid house data. It works fine
model0=getPickleModel('lidarJ0.dat')
#vtarget=points(pos=model0,color=(0,1,0))#green points
model1=getPickleModel('lidarJ1.dat')
#vmodel=points(pos=model1,color=(1,0,0))#red points
model2=getPickleModel('lidarJ2.dat')
#vmodel=points(pos=model2,color=(1,0,1))#purple points

#Grandfather house data not twisted. ICP doesn't work on it
# m0c=getPickleModel('lidar0c.dat')
# points(pos=m0c,color=(0,1,0))
# m1c=getPickleModel('lidar1c.dat')
# points(pos=m1c,color=(1,0,1))
# m2c=getPickleModel('lidar2c.dat')
# points(pos=m2c,color=(1,1,0))
# m3c=getPickleModel('lidar3c.dat')
# points(pos=m3c,color=(0,1,1))

#dataset to test ICP in 3D
model3D =np.array([(0,1,0),(1,1,0),(0,0,-1),(0,1,-1)])*500
points(pos=model3D ,color=(0,0,1))
Ti=np.array([[0.5,0.1,0.8],
             [0.1,0.5,0.3],
             [0.8,0.3,0.5]])
target3D=np.array([(0.1,1,0),(1.1,1.1,0),(0.1,0.1,-1),(0.1,1,-1)])*500
# target3D=np.array([[-103.47318056,   -4.73010521,  489.24986969],
#  [ -81.20086263 , 494.66670749,  499.58269659],
#  [-487.54454916 ,  24.66799255, -103.72983964],
#  [-591.94480715 ,  19.20743331  ,385.21879614]])
#target3D=Ti.dot(model3D.transpose()).transpose()+np.array([(1,1,1),(1,1,1),(1,1,1),(1,1,1)])*500

#dataset to fit the model in this target usually is a 2D map or a 3D geometric point cloud
target=model0
vtarget=points(pos=target,color=(0,1,0))

if __name__ == '__main__':
    Tf=np.eye(3,3)
    Tf,pit1=icp.fitICP(model1,target)
    #target=np.vstack((target,pit1))
    points(pos=pit1,color=(1,0,0))
    print Tf
    Tf,pit2=icp.fitICP(model2,target)
    points(pos=pit2,color=(1,0,1))
    #target=np.vstack((target,pit2))
    print Tf
    while True:
        rate(10)