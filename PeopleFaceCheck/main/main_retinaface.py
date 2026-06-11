#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/4/16 0016 15:29
# @Author : scw
# @File   : main_retinaface.py
# 使用RetinaFace进行人脸检测，CNN进行识别
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tfv1
import cv2
import os
import sys
from retinaface import RetinaFace
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

# 读取摄像头参数设为0(0为默认摄像头）
camera = cv2.VideoCapture(0)

print("RetinaFace模型首次运行会自动下载...")

while True:
    ret, img = camera.read()
    if not ret:
        continue

    # 使用RetinaFace检测人脸
    faces = RetinaFace.detect_faces(img)

    if len(faces) == 0:
        cv2.imshow('image', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
        continue

    # 遍历检测到的人脸
    for face_key in faces:
        face_data = faces[face_key]

        # 获取人脸区域坐标
        x1, y1, x2, y2 = face_data['facial_area']

        # 确保坐标在图像范围内
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img.shape[1], int(x2))
        y2 = min(img.shape[0], int(y2))

        # 提取人脸区域
        face = img[y1:y2, x1:x2]

        # 检查人脸区域是否有效
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            continue

        # 调整图片的尺寸
        face = cv2.resize(face, (size, size))

        # 用CNN识别
        name = name_dict[name_key(face)]

        # 绘制检测框和名字
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('image', img)
    key = cv2.waitKey(30) & 0xff
    # 当按了esc键之后，进行退出识别
    if key == 27:
        break

sess.close()
camera.release()
cv2.destroyAllWindows()
