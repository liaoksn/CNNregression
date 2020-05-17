import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt
#eigenVectorK=np.loadtxt('E:/ZJU study/training_data/Erecords/eigenVector.txt')
#eigenVectorK=eigenVectorK.reshape(-1,21300)
#print(eigenVectorK.shape)


#with open('E:/ZJU study/training_data/Erecords/images_list.txt','r') as fa:
#    with open('E:/ZJU study/training_data/Erecords/PointsFeature1588173121.txt','r') as fb:
#        with open('E:/ZJU study/training_data/Erecords/Ttrain_list.txt','w') as fc:
#            for line in fa:
#                fc.write(line.strip('\n'))
#                fc.write(' '+fb.readline())


# c=np.array([4, 3])
# d=np.array([1, 2])
# d=pow(c-d,2)
# print(d)

#loss_function=tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y, pred)))

#w1=tf.Variable(tf.random_normal([4, 15], stddev=1))
#w2=tf.Variable(tf.random_normal([4, 15], stddev=3))
#diff=tf.squared_difference(w1, w2)
#reduce_sum=tf.reduce_sum(diff,axis=0)
#reduce_mean=tf.reduce_mean(reduce_sum)
#epoch = tf.Variable(0, name='epoch', trainable=False)
#init_op = tf.global_variables_initializer()
#with tf.Session() as sess:
#for ep in range(0, 30):
    #with tf.Session() as sess:
#    sess.run(init_op)
#    for ep in range(30):
#        print("Train epoch:", '%02d' % (sess.run(epoch) + 1))
#        sess.run(epoch.assign(ep + 1))
#        print("epoch:", '%02d' % (sess.run(epoch)))

# etest=np.random.rand(3,2,15)
# #example = np.array([[[i for i in range(0, 5)],[0 for j in range(0, 5)]] for k in range(0, 10)])
#f = open('exampleData.csv', 'ab')
# restest=etest.reshape(-1,15)
#for i in etest:
#    print(i)
#    np.savetxt('E:/ZJU study/training_data/Erecords/testpred.txt',i, fmt="%f")
    #print('0')
    #np.savetxt(f, i, fmt='%i',delimiter=','
# print(etest)
# print('0')
# print(restest)
# fig = plt.gcf()
# fig.set_size_inches(4, 2)
# epoch_list=np.random.rand(15)
# loss_list=np.random.rand(15)
# plt.plot(epoch_list, loss_list, label='loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['loss'], loc='upper right')
# plt.show()