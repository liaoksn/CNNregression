import os

images_dir = 'E:/ZJU study/training_data/ExampleImage/Frontal_Images_ZIP'
#images_dir='E:/ZJU study/training_data/ExampleImage/images_data/Frontal_Images'
labels_dir='E:/ZJU study/training_data/Erecords/PointsFeature121.txt'
#images_list = 'E:/ZJU study/training_data/Erecords/Newimages_list.txt'
images_list = 'E:/ZJU study/training_data/Erecords/images_list.txt'
train_list='E:/ZJU study/training_data/Erecords/train_list.txt'

#生成图片名字列表，注意用listdir直接读取的文件排序不定，在图片名与obj名读入时均使用sort默认排序即可保证顺序
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
