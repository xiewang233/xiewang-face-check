# xiewang face check

基于 YOLOv8 + CNN 的人脸检测与识别系统

## 项目简介

本项目实现了一个完整的人脸检测与识别系统，采用 YOLOv8 进行人脸检测，结合自训练的 CNN 模型进行人脸识别。支持实时摄像头识别、图片识别、视频识别等多种应用场景。

## 功能特性

- **多种人脸检测方式**：支持 YOLOv8、RetinaFace、MediaPipe、OpenCV Haar
- **多种数据采集方式**：摄像头采集、图片采集、视频采集、网络采集
- **CNN 人脸识别**：基于 TensorFlow 的自定义卷积神经网络
- **实时识别**：支持摄像头实时人脸识别

## 项目结构

```
FaceCheckPython-master/
├── PeopleFaceCheck/
│   ├── main/                    # 主程序目录
│   │   ├── main_yolov8.py      # YOLOv8检测+CNN识别（推荐）
│   │   ├── main_retinaface.py  # RetinaFace检测+CNN识别
│   │   ├── main_mediapipe.py   # MediaPipe检测+CNN识别
│   │   ├── main_opencv.py      # OpenCV Haar检测+CNN识别
│   │   └── main.py             # 原始dlib检测+CNN识别
│   ├── get_faces/              # 人脸数据采集
│   │   ├── get_faces_from_camera.py    # 摄像头采集
│   │   ├── get_faces_from_photo.py     # 图片采集
│   │   ├── get_faces_from_video.py     # 视频采集
│   │   └── get_faces_from_Internet.py  # 网络采集
│   ├── CNN_training/          # CNN模型训练
│   │   ├── create_CNN_network.py       # CNN网络构建
│   │   └── train.py                   # 训练脚本
│   ├── training_material/      # 训练数据集
│   └── requirements.txt        # 依赖列表
├── model/                      # 训练好的模型文件
└── yolov8n.pt                 # YOLOv8模型文件
```

## 环境要求

- Python 3.5+
- OpenCV
- TensorFlow 1.x
- dlib
- NumPy
- scikit-learn
- ultralytics (YOLOv8)

## 安装步骤

### 1. 安装依赖

```bash
pip install -r PeopleFaceCheck/requirements.txt
pip install ultralytics
```

### 2. 下载 YOLOv8 模型

首次运行时程序会自动下载 `yolov8n.pt` 模型，或手动下载后放在项目根目录。

### 3. 准备训练数据

#### 方法一：使用摄像头采集

```bash
cd PeopleFaceCheck/get_faces
python get_faces_from_camera.py
```

按提示输入人名和采集照片数量，程序会自动采集人脸数据。

#### 方法二：使用现有图片

将人脸照片按以下结构存放：

```
training_material/
├── person1/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── person2/
│   ├── 1.jpg
│   └── ...
```

### 4. 训练 CNN 模型

```bash
cd PeopleFaceCheck/CNN_training
python train.py
```

训练完成后，模型会自动保存到 `model/` 目录。

### 5. 运行人脸识别

```bash
cd PeopleFaceCheck/main
python main_yolov8.py
```

按 `ESC` 键退出程序。

## 使用说明

### 人脸检测方式选择

| 检测方式 | 文件 | 特点 |
|---------|------|------|
| YOLOv8 | main_yolov8.py | 速度快，精度高，推荐 |
| RetinaFace | main_retinaface.py | 精度最高，速度较慢 |
| MediaPipe | main_mediapipe.py | 速度快，资源占用少 |
| OpenCV Haar | main_opencv.py | 无需额外依赖，传统方法 |

### 参数调整

在 `main_yolov8.py` 中可调整以下参数：

- `conf=0.5`：检测置信度阈值
- `size=64`：人脸图片尺寸
- `classes=0`：检测类别（0表示人）

## 技术架构

1. **人脸检测**：YOLOv8 目标检测模型
2. **人脸识别**：自定义 CNN 网络结构
   - 3个卷积层 + 池化层
   - 1个全连接层
   - Dropout 防止过拟合
3. **图像处理**：OpenCV
4. **深度学习框架**：TensorFlow 1.x

## 注意事项

1. 首次运行需要训练模型，请确保有足够的训练数据（建议每人100+张照片）
2. YOLOv8 模型会自动下载，请确保网络连接正常
3. 识别准确率受训练数据质量和数量影响
4. TensorFlow 1.x 已停止维护，建议后续迁移至 TensorFlow 2.x

## 许可证

本项目基于 MIT License 开源，详见 [LICENSE](LICENSE) 文件。

## 作者

邪王真翔bla

## 更新日志

- 2024：新增 YOLOv8、RetinaFace、MediaPipe 检测方式
- 2018：初始版本，基于 dlib + TensorFlow
