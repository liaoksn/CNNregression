import numpy as np
import os
import time  # 引入time模块

#注意此程序仅允许单个人体的文件输入
mappath='E:/ZJU study/training_data/Erecords/pred1.txt'
mapdata=np.loadtxt(mappath)
mapdata=mapdata.reshape(15,-1)
print(mapdata.shape)

eigenVectorK=np.loadtxt('E:/ZJU study/training_data/Erecords/eigenVector.txt')
print(eigenVectorK.shape)
#eigenVectorK=eigenVectorK.T
#print(eigenVectorK)
mpointspath='E:/ZJU study/training_data/Erecords/PointsMean121.txt'
pointsm=np.loadtxt(mpointspath)
pointsm=pointsm.reshape(21300,-1)
print(pointsm.shape)
points=np.dot(eigenVectorK,mapdata)+pointsm
points=points.T
points=points.reshape(-1,3)
print(points.shape)
prepointsp='E:/ZJU study/training_data/Erecords/Prepoints.txt'
np.savetxt(prepointsp,points,fmt="%f")