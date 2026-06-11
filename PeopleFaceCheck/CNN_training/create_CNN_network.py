#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/4/16 0016 15:32
# @Author : scw
# @File   : create_CNN_network.py
#此代码包含数据读入以及CNN神经网络的构建的主要函数

#导入主要函数库
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()  # 使用 TF1 行为
import tensorflow.compat.v1 as tfv1
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

# 设置tensorflow日志级别（可忽略）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 训练图片集位置
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training_material'))

# 图片规格统一64x64
size=64

# 初始化图像，标签列表
imgs=[]
labs=[]

# 字符串转整数
def str2num(init):
    num = 0
    for i in init:
        i = int(ord(i))
        num = num + i
    return num

# 计算path路径下的文件和文件夹数目
def count_dir(path):
    count = 0
    for dir in os.listdir(path):
        count += 1
    return count

# 计算数据集分类数dir_num
dir_num = count_dir(data_path)

# 获取图像填充大小
def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)
    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

# 生成标签列表way
way = []
for lab in os.listdir(data_path):
    way.append(data_path + '/' + lab)

# 读取图像信息
def readData(path, h=size, w=size):#获取单个文件夹下的图像信息
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top, bottom, left, right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(str2num(path))

# 从data_path下获取图像信息
for lab in os.listdir(data_path):
    readData(data_path + '/' + lab)

# 将标签排列成单位矩阵
def select_lab(lab,dir_num):
    a = []
    x = []
    for i in range(dir_num):
        x.append(0)
    for i in range(dir_num):
        x[i] = 1
        a.append(x[:])
        x[i] = 0
    for i in range(dir_num):
        if lab == str2num(way[i]):
            return a[i]

# 将图像与标签信息转化成矩阵形式
imgs = np.asarray(imgs, np.float32)
labs = np.array([select_lab(lab, dir_num) for lab in labs])

# 随机划分测试集与训练集
train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0,100))

# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

# 将数据转换成小于1的数
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

# 输出训练测试集数
print('train size:%s, test size:%s' % (len(train_x), len(test_x)))

# 每次取100张图片作为数据快
batch_size = 100
num_batch = len(train_x) // batch_size

# 设置占位符
x = tfv1.placeholder(tfv1.float32, [None, size, size, 3],name='x')
y_ = tfv1.placeholder(tfv1.int32, [None, dir_num],name='y_')

# 防止过拟合设置
keep_prob_5 = tfv1.placeholder(tfv1.float32)
keep_prob_75 = tfv1.placeholder(tfv1.float32)

# 设置CNN网络的相关构造函数
# 变权
def weightVariable(shape):
    init = tfv1.random_normal(shape, stddev=0.01)
    return tfv1.Variable(init)
# 偏差
def biasVariable(shape):
    init = tfv1.random_normal(shape)
    return tfv1.Variable(init)
# 卷积
def conv2d(x, W):
    return tfv1.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
# 池化
def maxPool(x):
    return tfv1.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# 一定概率丢弃数据，防止过拟合
def dropout(x, keep):
    return tfv1.nn.dropout(x, keep)

# 构建CNN神经网络
def cnnlayer():
    # 第一层
    # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    W1 = weightVariable([3,3,3,32])
    b1 = biasVariable([32])
    # 卷积
    conv1 = tfv1.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tfv1.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tfv1.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([8*8*64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tfv1.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,dir_num])
    bout = weightVariable([dir_num])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

# 训练函数
def cnnTrain():
    out = cnnlayer()
    cross_entropy = tfv1.reduce_mean(tfv1.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y_))
    train_step = tfv1.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tfv1.cast(强制转换类型)
    accuracy = tfv1.reduce_mean(tfv1.cast(tfv1.equal(tfv1.argmax(out, 1), tfv1.argmax(y_, 1)), tfv1.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tfv1.summary.scalar('loss', cross_entropy)
    tfv1.summary.scalar('accuracy', accuracy)
    merged_summary_op = tfv1.summary.merge_all()
    # 数据保存器的初始化
    saver = tfv1.train.Saver()
    with tfv1.Session() as sess:
        sess.run(tfv1.global_variables_initializer())
        summary_writer = tfv1.summary.FileWriter('./tmp', graph=tfv1.get_default_graph())
        for n in range(1000):
             # 每次取128(batch_size)张图片
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
                summary_writer.add_summary(summary, n*num_batch+i)
                # 打印损失
                print(n*num_batch+i, loss)
                if (n*num_batch+i) % 100 == 0:
                    # 获取测试数据的准确率
                    acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                    print(n*num_batch+i, acc)
                    # 准确率大于0.97时保存并退出,可修改
                    if acc > 0.98 and n > 2:
                        print("acc=%s"%acc)
                        model_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'train.model.ckpt'))
                        saver.save(sess, model_save_path, global_step=n*num_batch+i)#将模型保存在model下
                        sess.close()
                        sys.exit(0)
        print('accuracy less than %s, exited!'%accuracy)
