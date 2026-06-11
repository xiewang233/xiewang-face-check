#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/4/16 0016 15:29
# @Author : scw
# @File   : main_mediapipe.py
# 使用MediaPipe进行人脸检测，CNN进行识别
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tfv1
import cv2
import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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

# 创建MediaPipe人脸检测器
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=False, output_facial_transformation_matrixes=False, num_faces=5)
detector = vision.FaceLandmarker.create_from_options(options)

# 读取摄像头参数设为0(0为默认摄像头）
camera = cv2.VideoCapture(0)

print("MediaPipe人脸检测已启动...")

while True:
    ret, img = camera.read()
    if not ret:
        continue

    # 转换为RGB格式（MediaPipe需要RGB）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建MediaPipe图像对象
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # 检测人脸
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks is None or len(detection_result.face_landmarks) == 0:
        cv2.imshow('image', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
        continue

    h, w, _ = img.shape

    # 遍历检测到的人脸
    for face_landmarks in detection_result.face_landmarks:
        # 获取人脸边界框（从关键点计算）
        x_coords = [lm.x for lm in face_landmarks]
        y_coords = [lm.y for lm in face_landmarks]

        x1 = int(min(x_coords) * w)
        y1 = int(min(y_coords) * h)
        x2 = int(max(x_coords) * w)
        y2 = int(max(y_coords) * h)

        # 稍微扩展边界框
        margin = int((x2 - x1) * 0.2)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

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
