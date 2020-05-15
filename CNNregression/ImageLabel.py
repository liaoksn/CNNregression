import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

cat_filename = []
cat_label = []
dog_filename = []
dog_label = []
#tensorflow数据读取机制：包括创建文件名队列、读取文件放入文件队列中
#首先定义函数，将所有图片的文件名加入队列
def get_filename(file_dir):
    #获取猫和狗图片的地址
    #os.listdir：返回指定路径下的文件和文件夹列表。
    for filename in os.listdir(file_dir +"/Cat"):
        cat_filename.append(file_dir +"/Cat"+ "/"+ filename)
        #每添加一张猫的图片就在猫的标签列表中添加一个0，这里假定猫为0，狗为1
        cat_label.append(0)

    for filename in os.listdir(file_dir +"/Dog"):
        dog_filename.append(file_dir +"/Dog"+"/"+ filename)
        dog_label.append(1)

    #filenamelist和labellist都为一行的行向量
    filenamelist = np.hstack((cat_filename, dog_filename))
    labellist = np.hstack((cat_label, dog_label))

    return filenamelist, labellist

file_dir = "E:/ZJU study/training_data/ExampleImage"

#batch_size过小则训练数据会变得难以收敛，过大则内存容量增大。在这里为了演示方便设为2
BATCH_SIZE = 1

#capacity -- 队列容量
CAPACITY = 32

#image_Width = 256
#image_Length = 256

filenamelist, labellist = get_filename(file_dir)
filenamelist = tf.cast(filenamelist, tf.string)
labellist = tf.cast(labellist, tf.int32)

#生成文件名队列
#由于在本例中包括图片和标签所以用slice_input_producer，若只读取图像可以用string_input_producer
input_queue = tf.train.slice_input_producer([filenamelist, labellist])
#slice_input_producer(tensor_list, num_epochs=None, shuffle=True, seed=None,capacity=32, shared_name=None, name=None)

#读取和裁剪图像
image = tf.read_file(input_queue[0])
label = input_queue[1]

#image = tf.image.decode_jpeg(image, channels=3)
tf.image.decode_png(image,channels=0,dtype=tf.uint8,name=None)
#tf.image.decode_png(image,channels=0,dtype=tf.uint8,name=None) channels0：使用PNG编码图像中的通道数量.1：输出灰度图像.
#image = tf.image.resize_image_with_crop_or_pad(image, image_Length, image_Width)
#tf.image.resize_image_with_crop_or_pad:利用剪切或填充图像为目标大小

#生成文件队列
image_batch, label_batch = tf.train.batch([image, label], batch_size=BATCH_SIZE, num_threads=32, capacity=CAPACITY)
#tf.train.batch 按照给定的tensor顺序，把batch_size个tensor推送到文件队列，作为训练一个batch的数据

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()  #用于协调多个线程之间的关系

    # 只有调用 tf.train.start_queue_runners 之后，才会真正把tensor推入内存序列中，供计算单元调用，否则会由于内存序列为空，数据流图会处于一直等待状态。
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop() and i<5:
            img, label = sess.run([image_batch, label_batch])
            #np.arange(X),返回一个队列，内容为0-X
            for j in np.arange(BATCH_SIZE):
                #显示图片以及相应标签
                print('label: %d' %label[j])
                plt.imshow(img[j])
                plt.show()
            i+=1

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        #停止线程
        coord.request_stop()
    #把线程加入主线程，等待threads结束
    coord.join(threads)
    sess.close()