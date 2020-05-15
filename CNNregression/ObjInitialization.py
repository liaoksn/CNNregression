# 本模块作用是将obj文件中的点数据取出，写入pointsum矩阵，并进行一系列处理，最终结果为KXm特征系数矩阵；过程文件均保留
import numpy as np
import os
import time  # 引入time模块

ticks = time.time()

# 获取当前obj路径
read_path = 'E:/ZJU study/training_data/samples'
#read_path=os.path.join(work_path, 'objtest')


# 修改当前工作目录
os.chdir(read_path)

# 将该文件夹下的所有文件名存入列表
obj_name_list = os.listdir()
obj_name_list.sort()
# 建立用于装载point的矩阵
pointsum = np.zeros((21300, len(obj_name_list)))


for i in range(0, len(obj_name_list)):
    with open(obj_name_list[i]) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
                # points原本为列表，需要转变为矩阵，方便处理
        points = np.array(points)
        rpoints = points.reshape(21300)
        pointsum[:, i] = np.copy(rpoints)

print(pointsum.shape)

# 将pointsum中的数据保存在当前工作路径下，生成的一个名为OriginalPoints的文本文件中，该文件生成后此端程序无法再次运行
#np.savetxt('E:/ZJU study/training_data/Erecords/OriginalPoints{:.0f}.txt'.format(ticks),pointsum,fmt="%f")

# 求三维人体模型各点坐标均值(3nX1)并保存
PointMean=pointsum.mean(axis=1)
PointMean=np.array(PointMean).reshape(len(PointMean),1)
#np.savetxt('E:/ZJU study/training_data/Erecords/PointsMean{:.0f}.txt'.format(ticks),PointMean,fmt="%f")
#print(type(PointMean))
#print(PointMean.shape)

# 求各点减去均值后的残值,并保存
PointFeature=np.subtract(pointsum,PointMean)
#print("{}".format(PointFeature))
#np.savetxt('E:/ZJU study/training_data/Erecords/StandardPoints{:.0f}.txt'.format(ticks),PointFeature,fmt="%f")

# 输入前K个特征向量 (kX3n)
eigenVectorK=np.loadtxt('E:/ZJU study/training_data/Erecords/eigenVector.txt')
eigenVectorK=eigenVectorK.T
#print(eigenVectorK)
# 将残值矩阵映射到特征向量方向上得到各体型特征点构成的特征点矩阵(kX3n 3nXm),每个kX1数组即为一体型的特征
FeatureMatrix=np.matmul(eigenVectorK,PointFeature)
FeatureMatrix=FeatureMatrix.T
np.savetxt('E:/ZJU study/training_data/Erecords/PointsFeature{:.0f}.txt'.format(ticks),FeatureMatrix,fmt="%f")
print(FeatureMatrix.shape)
