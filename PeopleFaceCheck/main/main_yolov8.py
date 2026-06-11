

#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/4/16 0016 15:29
# @Author : scw
# @File   : main_yolov8.py
# 使用YOLOv8进行人脸检测，CNN进行识别
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tfv1
import cv2
import os
import sys
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

# 中文名映射（文件夹名 -> 显示的中文名）
chinese_name_map = {
    'jinboxiang': '靳博翔',
    'human': '陌生人'
}

# PIL绘制中文文字
def draw_text_cn(img, text, position, font_size, color):
    """在OpenCV图像上绘制中文文字"""
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 尝试使用系统中文字体
    font = None
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

    if font is None:
        font = ImageFont.load_default()

    # 绘制文字
    draw.text(position, text, font=font, fill=color)

    # 转换回OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

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

# 根据人体宽高比判断体型
def body_type(width, height):
    ratio = width / height
    if ratio > 0.45:
        return "Fat", (0, 0, 255)  # 红色
    elif ratio < 0.32:
        return "Thin", (255, 0, 255)  # 紫色
    else:
        return "Normal", (0, 255, 0)  # 绿色

# 使用YOLOv8进行人脸检测
# 使用专门的人脸检测模型
face_detector = YOLO('yolov8n.pt')

# 读取摄像头参数设为0(0为默认摄像头）
camera = cv2.VideoCapture(0)

# 全屏状态
fullscreen = False
window_name = 'xiewang face check'

# 创建窗口
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, img = camera.read()
    if not ret:
        continue

    # 使用YOLOv8检测人脸
    # conf=0.5 设置置信度阈值
    # classes=0 只检测"人"这个类别（COCO数据集中0是person）
    results = face_detector(img, conf=0.5, classes=0, verbose=False)

    # 获取检测结果
    boxes = results[0].boxes

    if len(boxes) == 0:
        img = draw_text_cn(img, '人数: 0', (10, 10), 40, (0, 255, 0))
        cv2.imshow(window_name, img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:  # ESC退出
            break
        elif key == ord('f'):  # F键切换全屏
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        continue

    # 显示检测到的人数
    person_count = len(boxes)
    img = draw_text_cn(img, f'人数: {person_count}', (10, 10), 40, (0, 255, 0))

    # 先收集所有要绘制的信息
    draw_info = []

    # 遍历检测到的人脸
    for box in boxes:
        # YOLOv8返回的是xyxy格式：x1, y1, x2, y2
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        # 提取人脸区域
        face = img[y1:y2, x1:x2]

        # 计算人体宽高并判断体型
        width = x2 - x1
        height = y2 - y1
        body_type_text, body_type_color = body_type(width, height)

        # 检查人脸区域是否有效
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            continue

        # 调整图片的尺寸
        face = cv2.resize(face, (size, size))

        # 用CNN识别
        folder_name = name_dict[name_key(face)]
        # 映射到中文名
        name = chinese_name_map.get(folder_name, folder_name)

        # 保存绘制信息
        draw_info.append({
            'box': (x1, y1, x2, y2),
            'name': name,
            'body_type': body_type_text,
            'body_color': body_type_color
        })

    # 绘制所有信息
    for info in draw_info:
        x1, y1, x2, y2 = info['box']
        # 绘制检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # 显示名字（上排）
        img = draw_text_cn(img, info['name'], (x1, y1 - 45), 35, (0, 0, 255))
        # 显示体型（框内底部）
        label_y = y2 - 35
        if label_y < y1 + 40:
            label_y = y1 + 45
        img = draw_text_cn(img, info['body_type'], (x1, label_y), 30, tuple(reversed(info['body_color'])))

    cv2.imshow(window_name, img)
    key = cv2.waitKey(30) & 0xff
    # 当按了esc键之后，进行退出识别
    if key == 27:  # ESC退出
        break
    elif key == ord('f'):  # F键切换全屏
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    elif key == ord('q'):  # Q键也可以退出
        break

sess.close()
camera.release()
cv2.destroyAllWindows()
