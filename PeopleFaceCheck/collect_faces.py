#!/usr/bin/python
# -*- coding:utf-8 -*-
# 人脸照片采集工具
import cv2
import os
from ultralytics import YOLO
import time

# 保存路径
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_material', '靳博翔'))
os.makedirs(save_dir, exist_ok=True)

# 加载YOLO检测器
print("正在加载YOLO模型...")
face_detector = YOLO('yolov8n.pt')

# 打开摄像头
camera = cv2.VideoCapture(0)

# 计数器
count = 0
target_count = 200  # 采集200张照片

print(f"\n=== 人脸采集工具 ===")
print(f"保存路径: {save_dir}")
print(f"目标数量: {target_count} 张")
print("\n操作说明:")
print("  - 看到检测框后，按 空格键 保存照片")
print("  - 按 ESC 键退出")
print("\n开始采集...\n")

last_saved_time = 0

while True:
    ret, img = camera.read()
    if not ret:
        continue

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

    # 显示进度
    progress = f"Collected: {count}/{target_count}"
    cv2.putText(img, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "Press SPACE to save, ESC to exit", (10, img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Face Collection - 靳博翔', img)

    key = cv2.waitKey(30) & 0xff

    # 按空格键保存照片
    if key == 32:  # 空格键
        if len(boxes) > 0 and count < target_count:
            # 获取第一张人脸
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
                current_time = time.time()
                if current_time - last_saved_time > 0.5:
                    print(f"已保存: {filename}")
                    last_saved_time = current_time

                if count >= target_count:
                    print(f"\n✓ 采集完成！共保存 {count} 张照片")
                    print(f"保存在: {save_dir}")
                    print("\n接下来运行训练脚本:")
                    print("  python PeopleFaceCheck/CNN_training/train.py")
                    break

    # 按ESC键退出
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()

if count > 0:
    print(f"\n采集结束，共保存 {count} 张照片")
else:
    print("\n未保存任何照片")
