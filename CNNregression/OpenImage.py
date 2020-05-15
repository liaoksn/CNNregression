import cv2 as cv
import numpy as np

def threshold_demo(image):                          #全局阈值
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
#    print("threshold value %s"%ret)
#    cv.imshow("global_threshold_binary", binary)
    return ret,binary

"""
cv2.threshold函数是有两个返回值的，
第一个返回值，得到图像的阈值，
第二个返回值，也就是阈值处理后的图像，
我们自己不一定能够找到一个最好的阈值，去二分化图像，所以我们需要算法自己去寻找一个阈值，而cv.THRESH_OTSU就可以满足这个需求，
去找到一个最好的阈值。
注意：他非常适用于图像灰度直方图具有双峰的情况，他会在双峰之间找到一个值作为阈值，对于非双峰图像，可能并不是很好用。
因为cv.THRESH_OTSU方法会产生一个阈值，那么函数cv2.threshold的的第二个参数（设置阈值）就是0（None）了，
并且在cv2.threshold的方法参数中还得加上语句cv2.THRESH_OTSU
这里面第三个参数maxval参数表示与THRESH_BINARY和THRESH_BINARY_INV阈值类型一起使用设置的最大值。
而我们使用的灰度图像最大则为255，所以设置为255即可
THRESH_OTSU最适用于双波峰 THRESH_TRIANGLE最适用于单个波峰，最开始用于医学分割细胞等
"""
"""def local_threshold(image):                       #局部阈值
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow("local_threshold_binary", binary)

def custom_threshold(image):        #自定义阈值
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w*h])  #降维
    mean = m.sum() / (w*h)    #求均值
   # print("mean : ", mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)#利用均值进行图像二值化
    cv.imshow("custom_threshold_binary", binary)
"""

src=cv.imread('D:/Pycharm/TFRecords1/images/1.jpg')
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
ret,globalBinary=threshold_demo(src)
cv.imwrite("1p.jpg", globalBinary)
cv.imshow("global_threshold_binary", globalBinary)

print(globalBinary[63,63])
#print(src.shape)

#local_threshold(src)
#custom_threshold(src)

cv.waitKey(0)
cv.destroyAllWindows()
