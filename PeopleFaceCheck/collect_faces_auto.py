#!/usr/bin/python
# -*- coding:utf-8 -*-
# 人脸照片自动采集工具
import cv2
import os
from ultralytics import YOLO
import time

# 保存路径
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_material', 'jinboxiang'))
os.makedirs(save_dir, exist_ok=True)

# 加载YOLO检测器
print("正在加载YOLO模型...")
face_detector = YOLO('yolov8n.pt')

# 打开摄像头
camera = cv2.VideoCapture(0)

# 计数器
count = 0
target_count = 200  # 采集200张照片

print(f"\n=== 人脸自动采集工具 ===")
print(f"保存路径: {save_dir}")
print(f"目标数量: {target_count} 张")
print("\n操作说明:")
print("  - 站在摄像头前，保持不同角度和表情")
print("  - 程序每0.3秒自动保存一张照片")
print("  - 按 ESC 键退出")
print("\n3秒后开始采集...\n")

# 倒计时
for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)

print("开始采集！请移动头部改变角度...\n")

last_save_time = 0
save_interval = 0.3  # 每0.3秒保存一张

while True:
    ret, img = camera.read()
    if not ret:
        continue

    current_time = time.time()

    # 检测人脸
    results = face_detector(img, conf=0.5, classes=0, verbose=False)
    boxes = results[0].boxes

    # 绘制检测框
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        # 提取人脸区域
        face = img[y1:y2, x1:x2]
        if face.size > 0 and face.shape[0] > 0 and face.shape[1] > 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 自动保存照片
            if len(boxes) > 0 and count < target_count and current_time - last_save_time > save_interval:
                x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy().astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)

                face = img[y1:y2, x1:x2]
                if face.size > 0 and face.shape[0] > 20 and face.shape[1] > 20:
                    # 调整大小为64x64
                    face = cv2.resize(face, (64, 64))
                    # 保存照片
                    filename = os.path.join(save_dir, f'face_{count}.jpg')
                    cv2.imwrite(filename, face)
                    count += 1
                    last_save_time = current_time

                    if count % 20 == 0:
                        print(f"已采集: {count}/{target_count} 张")

                    if count >= target_count:
                        print(f"\n✓ 采集完成！共保存 {count} 张照片")
                        print(f"保存在: {save_dir}")
                        print("\n接下来运行训练脚本:")
                        print("  python PeopleFaceCheck/CNN_training/train.py")
                        break

    # 显示进度
    progress = f"Collecting: {count}/{target_count}"
    cv2.putText(img, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "Move head! Auto-capturing...", (10, img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Auto Face Collection - jinboxiang', img)

    key = cv2.waitKey(30) & 0xff

    # 按ESC键退出
    if key == 27 or count >= target_count:
        break

camera.release()
cv2.destroyAllWindows()

if count > 0:
    print(f"\n采集结束，共保存 {count} 张照片")
else:
    print("\n未保存任何照片")
