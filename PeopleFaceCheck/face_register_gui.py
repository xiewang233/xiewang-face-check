#!/usr/bin/python
# -*- coding:utf-8 -*-
# 人脸信息录入工具 - GUI版本
import cv2
import os
import time
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 训练数据根目录
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_material'))
os.makedirs(data_root, exist_ok=True)

# 中文名映射文件
name_map_file = os.path.join(os.path.dirname(__file__), 'name_map.txt')
name_map = {}
if os.path.exists(name_map_file):
    with open(name_map_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=')
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

def draw_text_cn(img, text, position, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = None
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
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
    draw.text(position, text, font=font, fill=color)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

# 加载YOLO
print("加载YOLO模型...")
face_detector = YOLO('yolov8n.pt')
camera = cv2.VideoCapture(0)

# 状态变量
current_name = ""
temp_name = ""
collecting = False
count = 0
target_count = 100
last_save_time = 0
save_interval = 0.2
fullscreen = False

window_name = 'FaceRegister'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def reset_state():
    global current_name, temp_name, collecting, count
    current_name = ""
    temp_name = ""
    collecting = False
    count = 0

print("\n=== 人脸录入系统 ===\n")
print("操作说明:")
print("  直接输入姓名(拼音/英文)，按回车开始采集")
print("  采集中自动保存，按ESC完成")
print("  按F切换全屏，按Q退出\n")

# 预定义一些常用名字
preset_names = ["jinboxiang", "zhangsan", "lisi", "wangwu", "zhaoliu"]
current_preset_index = 0

while True:
    ret, img = camera.read()
    if not ret:
        continue

    display_img = img.copy()
    current_time = time.time()

    # 半透明面板
    overlay = display_img.copy()
    panel_height = 300
    cv2.rectangle(overlay, (0, 0), (img.shape[1], panel_height), (0, 0, 0), -1)
    display_img = cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0)

    if not collecting:
        # 输入模式
        display_img = draw_text_cn(display_img, '=== 人脸录入系统 ===',
                                   (50, 20), 40, (0, 255, 255))

        display_text = temp_name if temp_name else "输入姓名..."
        display_img = draw_text_cn(display_img, f'姓名: {display_text}_',
                                   (50, 80), 35, (255, 255, 0))

        # 快捷名字按钮
        display_img = draw_text_cn(display_img, '快捷键: 1-5选择预设名字',
                                   (50, 130), 22, (150, 200, 255))

        y_pos = 170
        for i, name in enumerate(preset_names):
            color = (0, 255, 0) if i == current_preset_index else (200, 200, 200)
            prefix = "> " if i == current_preset_index else "  "
            display_img = draw_text_cn(display_img, f'{prefix}{i+1}. {name}',
                                       (50, y_pos), 20, color)
            y_pos += 22

        # 已录入
        y_pos += 10
        display_img = draw_text_cn(display_img, f'已录入: {len(name_map)}人',
                                   (50, y_pos), 20, (0, 200, 255))

        # 操作提示
        tips = '输入|回车=开始|1-5=选预设|ESC=退出|Q=退出|T=训练'
        display_img = draw_text_cn(display_img, tips,
                                   (50, panel_height - 30), 18, (150, 150, 150))

    else:
        # 采集模式
        results = face_detector(img, conf=0.5, classes=0, verbose=False)
        boxes = results[0].boxes

        # 绘制检测框
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 自动保存
            if count < target_count and current_time - last_save_time > save_interval:
                face = img[y1:y2, x1:x2]
                if face.size > 0 and face.shape[0] > 20 and face.shape[1] > 20:
                    face = cv2.resize(face, (64, 64))
                    folder_name = chinese_to_folder(current_name)
                    person_dir = os.path.join(data_root, folder_name)
                    os.makedirs(person_dir, exist_ok=True)
                    filename = os.path.join(person_dir, f'face_{count}.jpg')
                    cv2.imwrite(filename, face)
                    count += 1
                    last_save_time = current_time

                    if count % 20 == 0:
                        print(f"[{current_name}] {count}/{target_count}")

                    if count >= target_count:
                        name_map[folder_name] = current_name
                        save_name_map()
                        print(f"完成: {current_name} - {count}张")
                        reset_state()

        # 显示信息
        display_img = draw_text_cn(display_img, f'录入: {current_name}',
                                   (50, 30), 35, (0, 255, 0))
        display_img = draw_text_cn(display_img, f'{count}/{target_count}',
                                   (400, 30), 35, (255, 255, 0))

        # 进度条
        bar_width = 500
        if count > 0:
            filled = int((count / target_count) * bar_width)
            cv2.rectangle(display_img, (50, 70), (50 + filled, 100), (0, 255, 0), -1)
        cv2.rectangle(display_img, (50, 70), (50 + bar_width, 100), (255, 255, 255), 2)

        display_img = draw_text_cn(display_img, '请转动头部!',
                                   (50, 130), 25, (255, 150, 0))

        display_img = draw_text_cn(display_img, 'ESC=完成',
                                   (50, panel_height - 30), 20, (255, 100, 100))

    cv2.imshow(window_name, display_img)
    key = cv2.waitKey(30) & 0xff

    # 全屏切换
    if key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    if not collecting:
        # 数字键选择预设名字
        if ord('1') <= key <= ord('5'):
            idx = key - ord('1')
            if idx < len(preset_names):
                temp_name = preset_names[idx]
                current_preset_index = idx

        if key == 13:  # Enter - 开始采集
            if temp_name.strip():
                current_name = temp_name.strip()
                collecting = True
                count = 0
                last_save_time = time.time()
                print(f"\n开始采集: {current_name}")
        elif key == 8:  # Backspace
            temp_name = temp_name[:-1]
        elif key == 27:  # ESC
            break
        elif key == ord('q'):  # Q退出
            break
        elif key == ord('t'):  # T训练
            print("\n开始训练模型...")
            camera.release()
            cv2.destroyAllWindows()
            import subprocess
            subprocess.run(['python', 'PeopleFaceCheck/CNN_training/train.py'])
            break
        elif 32 <= key <= 126:  # 可打印字符
            temp_name += chr(key)
    else:
        # 采集模式
        if key == 27:  # ESC完成
            if count > 10:
                folder_name = chinese_to_folder(current_name)
                name_map[folder_name] = current_name
                save_name_map()
                print(f"完成: {current_name} - {count}张")
            reset_state()
        elif key == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()

print("\n录入结束!")
print(f"总共录入: {len(name_map)} 人")
print("运行以下命令训练模型:")
print("  python PeopleFaceCheck/CNN_training/train.py")
