#!/usr/bin/python
# -*- coding:utf-8 -*-
# 使用YOLOv8进行人脸检测，CNN进行识别 - 优化版
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tfv1
import cv2
import os
import sys
import time
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import dlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CNN_training')))
import create_CNN_network as CNN

size = 64
CONFIDENCE_THRESHOLD = 0.6
INFER_SKIP = 2  # 每N帧运行一次CNN推理

# === 字体缓存 ===
_font_cache = {}

def _get_font(font_size):
    if font_size not in _font_cache:
        for path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]:
            if os.path.exists(path):
                try:
                    _font_cache[font_size] = ImageFont.truetype(path, font_size)
                    break
                except:
                    continue
        if font_size not in _font_cache:
            _font_cache[font_size] = ImageFont.load_default()
    return _font_cache[font_size]

def draw_texts_cn(img, texts):
    """批量绘制中文文字，只做一次PIL转换"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, pos, font_size, color in texts:
        draw.text(pos, text, font=_get_font(font_size), fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# === 构建名字映射 ===
def build_name_mapping():
    # 索引 -> 文件夹名
    dirs = sorted(os.listdir(CNN.data_path))
    index_to_folder = {i: d for i, d in enumerate(dirs)}
    # 为超出当前文件夹数量的类别索引填充默认名
    for i in range(CNN.dir_num):
        if i not in index_to_folder:
            index_to_folder[i] = f'class_{i}'
    index_to_folder[None] = 'no faces'

    # 文件夹名 -> 显示名（从name_map.txt加载）
    display_map = {'human': '陌生人'}
    name_map_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'name_map.txt'))
    if os.path.exists(name_map_file):
        with open(name_map_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    display_map[k] = v

    return index_to_folder, display_map

name_dict, chinese_name_map = build_name_mapping()

# === 构建CNN模型 ===
output = CNN.cnnlayer()
predict_op = tfv1.argmax(output, 1)
softmax_op = tf.nn.softmax(output)

saver = tfv1.train.Saver()
sess = tfv1.Session()
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
saver.restore(sess, tfv1.train.latest_checkpoint(model_path))

def recognize_face(face_img):
    """单次推理，返回 (预测ID, 置信度)"""
    feed = {CNN.x: [face_img / 255.0], CNN.keep_prob_5: 1.0, CNN.keep_prob_75: 1.0}
    pred, probs = sess.run([predict_op, softmax_op], feed_dict=feed)
    return pred[0], probs[0][pred[0]]

# === 体型判断 ===
def body_type(w, h):
    ratio = w / h
    if ratio > 0.45:
        return "Fat", (0, 0, 255)
    elif ratio < 0.32:
        return "Thin", (255, 0, 255)
    return "Normal", (0, 255, 0)

# === 初始化模型 ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
person_tracker = YOLO(os.path.join(project_root, 'yolov8n.pt'))
face_detector = dlib.get_frontal_face_detector()
camera = cv2.VideoCapture(0)

# === 状态变量 ===
fullscreen = debug_mode = False
window_name = 'Face Recognition - 邪王真翔'
track_name_map = {}
track_stranger_frames = {}
cached_results = {}
frame_count = 0
fps = 0.0
last_time = time.time()

register_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training_material', 'jinboxiang'))
next_face_idx = len(os.listdir(register_dir)) if os.path.exists(register_dir) else 0

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

while True:
    ret, img = camera.read()
    if not ret:
        continue

    # FPS计算
    now = time.time()
    dt = now - last_time
    last_time = now
    if dt > 0:
        fps = fps * 0.9 + (1.0 / dt) * 0.1
    frame_count += 1
    do_infer = (frame_count % INFER_SKIP) == 0

    results = person_tracker.track(img, conf=0.5, classes=0, verbose=False, persist=True)
    boxes = results[0].boxes

    texts = []  # 收集所有需要绘制的文字

    if len(boxes) == 0:
        texts.append((f'人数: 0  FPS: {fps:.0f}', (10, 10), 40, (0, 255, 0)))
        img = draw_texts_cn(img, texts)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                 cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        continue

    person_count = len(boxes)
    texts.append((f'人数: {person_count}  FPS: {fps:.0f}', (10, 10), 40, (0, 255, 0)))

    draw_list = []

    for box in boxes:
        px1, py1, px2, py2 = box.xyxy[0].cpu().numpy().astype(int)
        tid = int(box.id.cpu().numpy()) if box.id is not None else None

        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(img.shape[1], px2), min(img.shape[0], py2)

        bt_text, bt_color = body_type(px2 - px1, py2 - py1)

        # 人脸检测与识别（帧跳过优化）
        rec_name = '陌生人'
        confidence = 0.0
        face_found = False
        face_resized = None
        pred_id = None
        folder_name = None

        if do_infer or tid not in cached_results:
            roi = img[py1:py2, px1:px2]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                face_rects = face_detector(gray, 1)
                if face_rects:
                    d = face_rects[0]
                    fx1 = px1 + max(0, d.left())
                    fy1 = py1 + max(0, d.top())
                    fx2 = px1 + min(roi.shape[1], d.right())
                    fy2 = py1 + min(roi.shape[0], d.bottom())

                    if fx2 > fx1 and fy2 > fy1:
                        face_crop = img[fy1:fy2, fx1:fx2]
                        if face_crop.size > 0:
                            face_resized = cv2.resize(face_crop, (size, size))
                            pred_id, confidence = recognize_face(face_resized)
                            folder_name = name_dict.get(pred_id)
                            if confidence >= CONFIDENCE_THRESHOLD:
                                rec_name = chinese_name_map.get(folder_name, folder_name)
                            else:
                                rec_name = '陌生人'
                            face_found = True

            if tid is not None:
                cached_results[tid] = {
                    'name': rec_name, 'conf': confidence,
                    'found': face_found, 'pid': pred_id,
                    'folder': folder_name, 'face': face_resized
                }
        elif tid in cached_results:
            c = cached_results[tid]
            rec_name, confidence, face_found = c['name'], c['conf'], c['found']
            pred_id, folder_name, face_resized = c['pid'], c['folder'], c['face']

        # 追踪ID名字一致性
        if tid is not None:
            if tid in track_name_map:
                prev = track_name_map[tid]
                if rec_name != '陌生人':
                    track_name_map[tid] = rec_name
                    track_stranger_frames[tid] = 0
                    name = rec_name
                else:
                    track_stranger_frames[tid] = track_stranger_frames.get(tid, 0) + 1
                    name = prev if track_stranger_frames[tid] < 10 else rec_name
            else:
                track_name_map[tid] = rec_name
                track_stranger_frames[tid] = 0
                name = rec_name
        else:
            name = rec_name

        if not face_found:
            name = '未检测到人脸'

        draw_list.append({
            'box': (px1, py1, px2, py2), 'name': name,
            'conf': confidence if face_found else 0,
            'bt': bt_text, 'bt_color': bt_color,
            'face': face_resized, 'tid': tid,
            'dbg': (pred_id, folder_name, confidence) if face_found else None
        })

    # 绘制所有检测框和文字
    for d in draw_list:
        x1, y1, x2, y2 = d['box']
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        label = (d['name'] or '未知') + (f" {d['conf']:.0%}" if d['conf'] > 0 else "")
        texts.append((label, (x1, y1 - 45), 35, (0, 0, 255)))
        ly = max(y2 - 35, y1 + 45)
        texts.append((d['bt'], (x1, ly), 30, tuple(reversed(d['bt_color']))))

        if debug_mode and d['dbg']:
            pid, folder, conf = d['dbg']
            cv2.putText(img, f"ID:{pid} {folder} {conf:.2f}", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 批量绘制所有中文文字（单次PIL转换）
    img = draw_texts_cn(img, texts)

    cv2.imshow(window_name, img)
    key = cv2.waitKey(30) & 0xff

    if key == 27:
        break
    elif key == ord('f'):
        fullscreen = not fullscreen
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                             cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"调试模式: {'开启' if debug_mode else '关闭'}")
    elif key == ord('r'):
        if draw_list and draw_list[0]['face'] is not None:
            sp = os.path.join(register_dir, f'face_{next_face_idx}.jpg')
            cv2.imwrite(sp, draw_list[0]['face'])
            print(f'已保存人脸到: {sp}')
            next_face_idx += 1
        else:
            print('未检测到人脸，无法保存')
    elif key == ord('q'):
        break

sess.close()
camera.release()
cv2.destroyAllWindows()
