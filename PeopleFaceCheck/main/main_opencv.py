
#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/4/16 0016 15:29
# @Author : scw
# @File   : main_opencv.py
# 使用OpenCV Haar Cascade进行人脸检测，CNN进行识别
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tfv1
import cv2
import os
import sys
# 添加CNN生成代码的路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CNN_training')))
import create_CNN_network as CNN

size = 64

# 生成索引字典
def named_dict(path):
    dirs=os.listdir(path)
    name_dict={}
    test={}
    for i in range(len(dirs)):
        test={i:dirs[i]}
        name_dict.update(test)
        test.clear()
    test={None:'no faces'}
    name_dict.update(test)
    return name_dict

name_dict = named_dict(CNN.data_path)

output = CNN.cnnlayer()
predict = tfv1.argmax(output, 1)

saver = tfv1.train.Saver()
sess = tfv1.Session()
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
saver.restore(sess, tfv1.train.latest_checkpoint(model_path))

# 获取key
def name_key(image):
    res = sess.run(predict, feed_dict={CNN.x: [image / 255.0], CNN.keep_prob_5: 1.0, CNN.keep_prob_75: 1.0})[0]
    return res

# 使用OpenCV的Haar Cascade人脸检测器
# 模型文件包含在cv2.data中，会自动加载
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取摄像头参数设为0(0为默认摄像头）
camera = cv2.VideoCapture(0)

print("OpenCV Haar Cascade人脸检测已启动...")

while True:
    ret, img = camera.read()
    if not ret:
        continue

    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        cv2.imshow('image', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
        continue

    # 遍历检测到的人脸
    for (x, y, w, h) in faces:
        # 提取人脸区域
        face = img[y:y+h, x:x+w]

        # 检查人脸区域是否有效
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            continue

        # 调整图片的尺寸
        face = cv2.resize(face, (size, size))

        # 用CNN识别
        name = name_dict[name_key(face)]

        # 绘制检测框和名字
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('image', img)
    key = cv2.waitKey(30) & 0xff
    # 当按了esc键之后，进行退出识别
    if key == 27:
        break

sess.close()
camera.release()
cv2.destroyAllWindows()
