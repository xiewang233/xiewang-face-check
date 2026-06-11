#!/usr/bin/python
# -*- coding:utf-8 -*-
# 人脸信息录入工具 - 自动采集模式
import cv2
import os
import time
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 训练数据根目录
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_material'))
os.makedirs(data_root, exist_ok=True)

# 项目根目录（yolov8n.pt所在位置）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 中文名映射文件
name_map_file = os.path.join(os.path.dirname(__file__), 'name_map.txt')
name_map = {}
if os.path.exists(name_map_file):
    with open(name_map_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                name_map[k] = v

def save_name_map():
    with open(name_map_file, 'w', encoding='utf-8') as f:
        for folder, chinese in name_map.items():
            f.write(f'{folder}={chinese}\n')

def chinese_to_folder(name):
    try:
        from pypinyin import lazy_pinyin
        return ''.join(lazy_pinyin(name))
    except:
        return f"person_{int(time.time())}"

# 字体缓存
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
    """批量绘制中文文字"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, pos, font_size, color in texts:
        draw.text(pos, text, font=_get_font(font_size), fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def collect_faces(chinese_name, num_photos=100):
    folder_name = chinese_to_folder(chinese_name)

    person_dir = os.path.join(data_root, folder_name)
    os.makedirs(person_dir, exist_ok=True)

    name_map[folder_name] = chinese_name
    save_name_map()

    print("加载YOLO模型...")
    face_detector = YOLO(os.path.join(project_root, 'yolov8n.pt'))

    camera = cv2.VideoCapture(0)

    count = 0
    last_save_time = 0
    save_interval = 0.2

    window_name = f'录入: {chinese_name}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"\n开始采集 {chinese_name} 的照片...")
    print("请转动头部改变角度和表情")
    print("按ESC完成采集\n")

    while True:
        ret, img = camera.read()
        if not ret:
            continue

        current_time = time.time()
        texts = []

        results = face_detector(img, conf=0.5, classes=0, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if count < num_photos and current_time - last_save_time > save_interval:
                face = img[y1:y2, x1:x2]
                if face.size > 0 and face.shape[0] > 20 and face.shape[1] > 20:
                    face = cv2.resize(face, (64, 64))
                    filename = os.path.join(person_dir, f'face_{count}.jpg')
                    cv2.imwrite(filename, face)
                    count += 1
                    last_save_time = current_time

                    if count % 20 == 0:
                        print(f"已采集: {count}/{num_photos}")

                    if count >= num_photos:
                        print(f"\n{chinese_name} 采集完成！共{count}张")
                        time.sleep(1)
                        camera.release()
                        cv2.destroyAllWindows()
                        return True

        texts.append((f'正在录入: {chinese_name}', (50, 50), 50, (0, 255, 0)))
        texts.append((f'进度: {count}/{num_photos}', (50, 120), 40, (255, 255, 0)))

        bar_width = 600
        bar_height = 40
        if count > 0:
            filled = int((count / num_photos) * bar_width)
            cv2.rectangle(img, (50, 180), (50 + filled, 180 + bar_height), (0, 255, 0), -1)
        cv2.rectangle(img, (50, 180), (50 + bar_width, 180 + bar_height), (255, 255, 255), 2)

        texts.append(('请转动头部改变角度...', (50, 260), 30, (200, 200, 255)))
        texts.append(('按ESC完成采集', (50, img.shape[0] - 50), 30, (255, 100, 100)))

        img = draw_texts_cn(img, texts)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(30) & 0xff

        if key == 27:
            print(f"\n{chinese_name} 采集结束，共{count}张")
            camera.release()
            cv2.destroyAllWindows()
            return count >= 10
        elif key == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
            return False

def main():
    print("\n" + "="*50)
    print("           人脸信息录入系统")
    print("="*50)

    while True:
        print("\n已录入人员:")
        if name_map:
            for folder, chinese in name_map.items():
                print(f"  - {chinese}")
        else:
            print("  (暂无)")

        print("\n选项:")
        print("  1. 录入新人员")
        print("  2. 退出并训练模型")

        choice = input("\n请选择 (1/2): ").strip()

        if choice == '1':
            chinese_name = input("请输入姓名: ").strip()
            if chinese_name:
                collect_faces(chinese_name, num_photos=100)
        elif choice == '2':
            print("\n录入结束！")
            if name_map:
                print(f"\n共录入 {len(name_map)} 人")
                print("\n接下来运行训练脚本:")
                print("  python PeopleFaceCheck/CNN_training/train.py")
            break

if __name__ == '__main__':
    main()
