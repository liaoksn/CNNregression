import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
import numpy as np

train_record_list = 'E:/ZJU study/training_data/Erecords/train.tfrecords'
resize_height = 264  # 指定存储图片高度
resize_width = 192  # 指定存储图片宽度
tf.reset_default_graph()

# 1.1定义共享函数
# 定义权值
def weight(shape):
    # 构建模型时使用tf.Variable创建的变量
    # 使用函数tf.truncated_normal(截断的正态分布)生成标准差为0.1的随机数初始化权值
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')


# 定义偏置并初始化为0.1
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')


# 定义卷积操作
# 步长为1，padding为‘SAME’，即保持卷积后后大小与之前一致
def conv2d(x, W):
    # tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None,name=None)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化操作，此处的padding等于SAME表示在剩余位置不足时自动补0后池化
#ksize及strides中的[]：[batch, height, width, channels]
def max_pool_3x3(x):
    # tf.nn.max_pool(value,ksize,strides,padding,name=None)
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

def max_pool_4x4(x):
    # tf.nn.max_pool(value,ksize,strides,padding,name=None)
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 3, 3, 1], padding='SAME')

def max_pool_5x5(x):
    # tf.nn.max_pool(value,ksize,strides,padding,name=None)
    return tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1], padding='VALID')

# 1.2定义网络结构
# 输入层 264x192x3
with tf.name_scope('input_layer'):
    x = tf.placeholder('float', shape=[None, 264, 192, 3], name="x")

# 第1个卷积层
# 输入通道：3，输出通道：48，卷积后图像尺寸不变，依然是264x192
with tf.name_scope('conv_1'):
    W1 = weight([1, 1, 3, 48])  # [k_width,k_height,input_chn,output_chn]
    b1 = bias([48])  # 与output_chn一致
    conv_1 = conv2d(x, W1) + b1
    conv_1 = tf.nn.relu(conv_1)

# 第1个池化层
# 将264x192图像缩小为88X64，池化不改变通道数量，因此依然是48个
with tf.name_scope('pool_1'):
    pool_1 = max_pool_3x3(conv_1)

# 第2个卷积层
# 输入通道：48，输出通道：128，卷积后图像尺寸不变，依然是88X64
with tf.name_scope('conv_2'):
    W2 = weight([5, 5, 48, 128])  # [k_width,k_height,input_chn,output_chn]
    b2 = bias([128])  # 与output_chn一致
    conv_2 = conv2d(pool_1, W2) + b2
    conv_2 = tf.nn.relu(conv_2)

# 第2个池化层
# 将88x64图像缩小为29x21，池化不改变通道数量，因此依然是128个
with tf.name_scope('pool_2'):
    pool_2 = max_pool_4x4(conv_2)

# 第3个卷积层
# 输入通道：128，输出通道：192，卷积后图像尺寸不变，依然是29x21
with tf.name_scope('conv_3'):
    W3 = weight([3, 3, 128, 192])  # [k_width,k_height,input_chn,output_chn]
    b3 = bias([192])  # 与output_chn一致
    conv_3 = conv2d(pool_2, W3) + b3
    conv_3 = tf.nn.relu(conv_3)

# 第4个卷积层
# 输入通道：192，输出通道：192，卷积后图像尺寸不变，依然是29x21
with tf.name_scope('conv_4'):
    W4 = weight([3, 3, 192, 192])  # [k_width,k_height,input_chn,output_chn]
    b4 = bias([192])  # 与output_chn一致
    conv_4 = conv2d(conv_3, W4) + b4
    conv_4 = tf.nn.relu(conv_4)

# 第5个卷积层
# 输入通道：192，输出通道：128，卷积后图像尺寸不变，依然是29x21
with tf.name_scope('conv_5'):
    W5 = weight([3, 3, 192, 128])  # [k_width,k_height,input_chn,output_chn]
    b5 = bias([128])  # 与output_chn一致
    conv_5 = conv2d(conv_4, W5) + b5
    conv_5 = tf.nn.relu(conv_5)

# 第三个池化层，将图像转化为7X5
with tf.name_scope('pool_3'):
    pool_3 = max_pool_5x5(conv_5)

# 全连接层
# 将第3个池化层的128个7X5的图像转换为一维向量，长度为128*5*7=4480
# 定义为4096个神经元，可修改

with tf.name_scope('fc1'):
    W6 = weight([4480, 2048])  # 4096个神经元
    b6 = bias([2048])
    flat = tf.reshape(pool_3, [-1, 4480])
    h1 = tf.nn.relu(tf.matmul(flat, W6) + b6)
    h1_dropout = tf.nn.dropout(h1, keep_prob=0.8)


with tf.name_scope('fc2'):
    W7 = weight([2048, 128])  # 2048个神经元
    b7 = bias([128])
    h2 = tf.nn.relu(tf.matmul(h1_dropout, W7) + b7)
    h2_dropout = tf.nn.dropout(h2, keep_prob=0.8)

# 输出层，共15个神经元，分别对应15个标签值
with tf.name_scope('output_layer'):
    W8 = weight([128, 15])
    b8 = bias([15])
    #pred= tf.nn.relu(tf.matmul(h2_dropout, W8) + b8)
    pred = tf.matmul(h2_dropout, W8) + b8

# 1.3构建模型
with tf.name_scope("optimizer"):
    # 定义占位符
    y = tf.placeholder("float", shape=[None, 15], name="label")
    # 定义均方差损失函数！！！该表达式是否正确？
    #loss_function = tf.reduce_mean(tf.square(y - pred))
    loss_function=tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y, pred),axis=1))
    #rediction=tf.Variable('float', shape=[None, 15],, name='predic')

    #tf.squared_difference:计算张量 x、y 对应元素差平方
    #tf.reduce_sum() 用于计算张量tensor沿着某一维度的和,默认计算所有元素总和
    #tf.reduce_mean(Tensor)：降维平均，类似tf.reduce_sum()

    # 选择优化器
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.0001).minimize(loss_function)

# 1.4定义准确率!!!思考我所用的准确率该如何表达
#对回归问题，应使用mse评价
#with tf.name_scope("evaluation"):
#   correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#1.5 定义TFrecords读取相关函数
def read_records(filename, resize_height, resize_width, type=None):
    '''
    读入单个TFrecord
    解析record文件:源文件的图像数据是RGB,uint8,[0,255],一般作为训练数据时,需要归一化到[0,1]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param type:选择图像数据的返回类型
         None:默认将uint8-[0,255]转为float32-[0,255]
         normalization:归一化float32-[0,1]
         standardization:归一化float32-[0,1],再减均值中心化
    :return:
    '''
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'labels': tf.FixedLenFeature([], tf.string)
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)  # 获得图像原始的数据

    #tf_height = features['height']
    #tf_width = features['width']
    #tf_depth = features['depth']

    tf_label = tf.decode_raw(features['labels'], tf.float32)
    # PS:恢复原始图像数据,reshape的大小必须与保存之前的图像shape一致,否则出错
    tf_image = tf.reshape(tf_image, [resize_height, resize_width, 3])  # 设置图像的维度
    tf_label = tf.reshape(tf_label, [15])  # 设置图像label的维度

    # 恢复数据后,才可以对图像进行resize_images:输入uint->输出float32
    # tf_image=tf.image.resize_images(tf_image,[224, 224])

    # [3]数据类型处理
    # 存储的图像类型为uint8,tensorflow训练时数据必须是tf.float32
    if type is None:
        tf_image = tf.cast(tf_image, tf.float32)
    elif type == 'normalization':  # [1]若需要归一化请使用:
        # 仅当输入数据是uint8,才会归一化[0,255]
        # tf_image = tf.cast(tf_image, dtype=tf.uint8)
        # tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255.0)  # 归一化
    elif type == 'standardization':  # 标准化
        # 若需要归一化,且中心化,假设均值为0.5,请使用:
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5  # 中心化

    # 这里仅仅返回图像和标签
    return tf_image, tf_label

def get_batch_images(images, labels, batch_size, labels_nums, one_hot=False, shuffle=False, num_threads=1):
    '''
    :param images:图像
    :param labels:标签
    :param batch_size:
    :param labels_nums:标签个数
    :param one_hot:是否将labels转为one_hot的形式
    :param shuffle:是否打乱顺序,一般train时shuffle=True,验证时shuffle=False
    :return:返回batch的images和labels
    '''
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size  # 保证capacity必须大于min_after_dequeue参数值
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images, labels],
                                                            batch_size=batch_size,
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue,
                                                            num_threads=num_threads)
    else:
        images_batch, labels_batch = tf.train.batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    num_threads=num_threads)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch, labels_batch

def get_example_nums(tf_records_filenames):
    '''
    统计tf_records图像的个数(example)个数
    :param tf_records_filenames: tf_records文件路径
    :return:
    '''
    nums = 0
    for record in tf.python_io.tf_record_iterator(tf_records_filenames):
        nums += 1
    return nums

# 训练模型
# 2.1启动会话

train_epochs = 30
batch_size = 2  #？？？思考后再确定及XY数据如何输入
Xtrain = get_example_nums(train_record_list)
total_batch = int(Xtrain/ batch_size)
epoch_list = []
#accuracy_list = []
loss_list = []
pred_list=[]

epoch = tf.Variable(0, name='epoch', trainable=False)
startTime = time()
#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)


# 2.2断点续训
# 设置检查点存储目录
ckpt_dir = "E:/ZJU study/training_data/Erecords/AlexTrain_log/train_log"    #存储目录待确定
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# 生成saver
saver = tf.train.Saver(max_to_keep=5)

# 如果有检查点文件，读取最新的检查点文件，恢复各种变量值
ckpt = tf.train.latest_checkpoint(ckpt_dir)
#if ckpt != None:
#    saver.restore(sess, ckpt)  # 加载所有参数
#    # 从这里开始可直接使用模型进行预测，或者接着训练
#else:
#    print("Training from scratch.")

# 获取续训参数

#start = sess.run(epoch)
#print("Training starts from {} epoch.".format(start + 1))




# 2.3迭代训练
'''
def get_train_batch(number, batch_size):
     
    #return Xtrain[number * batch_size:(number + 1) * batch_size], Ytrain[number * batch_size:(number + 1) * batch_size]
    image_batch, label_batch = get_batch_images(tf_image, tf_label, batch_size=4, labels_nums=15, one_hot=False,
                                                shuffle=True)
    return 
'''

#for ep in range(start, train_epochs):
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    sess.run(local_init_op)
    sess.run(init)
    if ckpt != None:
        saver.restore(sess, ckpt)  # 加载所有参数
        # 从这里开始可直接使用模型进行预测，或者接着训练
    else:
        print("Training from scratch.")
    start = sess.run(epoch)
    print("Training starts from {} epoch.".format(start + 1))
    #with tf.Session() as sess:
    for ep in range(start, train_epochs):
        tf_image, tf_label = read_records(train_record_list, resize_height, resize_width, type='normalization')
        image_batch, label_batch = get_batch_images(tf_image, tf_label, batch_size=batch_size, labels_nums=15, one_hot=False,
                                                    shuffle=True)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                for i in range(total_batch):
                    batch_x, batch_y = sess.run([image_batch, label_batch])
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

                    if i % 5 == 0:
                        print("Step {}".format(i), "finished")

                loss = sess.run(loss_function, feed_dict={x: batch_x, y: batch_y})
                epoch_list.append(ep + 1)
                loss_list.append(loss)

                print("Train epoch:", '%02d' % (sess.run(epoch) + 1), "Loss=", "{:.6f}".format(loss))

                test_pred = sess.run(pred, feed_dict={x: batch_x, y: batch_y})  #输出预测值
                pred_list.append(test_pred)
                #print(test_pred)
                np_pred = np.array(pred_list)
                #print(np_pred.shape)!!!
                np_pred=np_pred.reshape(-1,15)
                np.savetxt('E:/ZJU study/training_data/Erecords/pred{}.txt'.format(ep), np_pred,fmt="%f")

                # 保存检查点
                saver.save(sess, ckpt_dir +'/'+ "hb_cnn_model.cpkt", global_step=ep + 1)
                sess.run(epoch.assign(ep + 1))
        except tf.errors.OutOfRangeError:
            print('Done training for epochs')
        finally:
        # Stop the threads
            coord.request_stop()
            # Wait for threads to stop
            coord.join(threads)


duration = time() - startTime
print("Train finished takes:", duration)

# 2.4可视化损失值

fig = plt.gcf()
fig.set_size_inches(4, 2)
plt.plot(epoch_list, loss_list, label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
