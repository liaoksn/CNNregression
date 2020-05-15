import os
#import numpy as np

#images_to_ids = {}
#file = []
#j = 0
#ksize=15
images_dir = 'E:/ZJU study/training_data/ExampleImage/Frontal_Images_ZIP'
#images_dir='E:/ZJU study/training_data/ExampleImage/images_data/Frontal_Images'
labels_dir='E:/ZJU study/training_data/Erecords/PointsFeature121.txt'
#images_list = 'E:/ZJU study/training_data/Erecords/Newimages_list.txt'
images_list = 'E:/ZJU study/training_data/Erecords/images_list.txt'
train_list='E:/ZJU study/training_data/Erecords/train_list.txt'

#生成图片名字列表，注意用listdir直接读取的文件排序不定，但sort升序按某字母排序与系统自带升序不一致，目前该问题尚未解决
fd = open(images_list, 'w')
image_names=os.listdir(images_dir)
image_names.sort()
for image_name in image_names:
    fd.write('{}\n'.format(image_name))
fd.close()


#打开图片名文件及对应标签文件，将图片文件的每一行末尾换行符去掉后再与标签文件行连接，写入训练数据文件
with open(images_list,'r') as fa:
    with open(labels_dir,'r') as fb:
        with open(train_list,'w') as fc:
            for line in fa:
                fc.write(line.strip('\n'))
                fc.write(' '+fb.readline())


#dirs记录正在遍历的文件夹下的子文件夹集合；files：记录正在遍历的文件夹中的文件集合
#for root, dirs, files in os.walk(data_dir):
#    file = dirs
#    break

#从txt文件中读取标签
#labels=np.loadtxt(labels_dir)

#for i in file:
#    class_names_to_ids.setdefault(i,count)
#    count = count + 1
#print(class_names_to_ids)
#print("total: ",count," class")

#for file in os.listdir(data_dir):
#    images_to_ids.setdefault(file, labels[j,:])
#    j=j+1
    #字典的keys，返回该字典中索引的名字
#for class_name in class_names_to_ids.keys():
	#os.listdir：返回指定路径下的文件和文件夹列表。
#    images_list = os.listdir(data_dir +'/'+ class_name)
#    for image_name in images_list:
#        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))
#fd.close()
