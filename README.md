# FaceCheckPython - 人脸检测与识别系统

一个基于 **YOLOv8** 和 **CNN** 的实时人脸识别方案，支持多人追踪、中文显示、体型分析等功能。

---

## 功能特性

- **多人实时识别** — YOLOv8 多目标追踪，每个人分配独立 ID
- **CNN 人脸识别** — 3 层卷积神经网络，支持动态类别数
- **中文显示** — 基于 PIL 的中文文字渲染，无乱码
- **体型分析** — 根据人体宽高比判断胖/标准/瘦
- **自动采集** — `face_register.py` 自动采集人脸训练数据，支持中文姓名
- **实时注册** — 识别过程中按 `R` 键即时保存人脸
- **全屏模式** — 按 `F` 键切换

### 快捷键

| 按键 | 功能 |
|------|------|
| `ESC` / `Q` | 退出程序 |
| `F` | 切换全屏/窗口 |
| `D` | 切换调试模式（显示预测 ID、置信度） |
| `R` | 保存当前人脸到训练数据 |

---

## 技术架构

```
摄像头 → YOLOv8 (人体检测+追踪) → dlib (人脸定位) → CNN (人脸识别) → 结果显示
```

### CNN 网络结构

```
输入: 64x64x3
  → Conv1 (3x3x32) + ReLU + MaxPool + Dropout(0.5)
  → Conv2 (3x3x64) + ReLU + MaxPool + Dropout(0.5)
  → Conv3 (3x3x64) + ReLU + MaxPool + Dropout(0.5)
  → FC (4096 → 512) + Dropout(0.75)
  → Output (512 → N classes)
```

---

## 项目结构

```
FaceCheckPython-master/
├── yolov8n.pt                              # YOLOv8 预训练模型
├── model/                                  # 训练检查点（根目录）
│   ├── train.model.ckpt-400.*
│   └── ...
│
└── PeopleFaceCheck/
    ├── main/
    │   └── main_yolov8.py                  # 主程序（运行这个）
    ├── CNN_training/
    │   ├── create_CNN_network.py           # CNN 网络定义 + 训练逻辑
    │   └── train.py                        # 训练入口
    ├── face_register.py                    # 人脸数据采集工具
    ├── name_map.txt                        # 文件夹名 → 中文姓名映射
    ├── requirements.txt                    # Python 依赖
    ├── model/
    │   ├── train.model.ckpt-*.meta/index/data  # CNN 模型文件
    │   ├── checkpoint                      # 指向最新检查点
    │   └── num_classes.txt                 # 模型类别数（训练时自动生成）
    └── training_material/                  # 训练数据集
        ├── human/                          # 陌生人样本
        └── <姓名拼音>/                     # 各人员的人脸照片
```

---

## 环境要求

- **OS**: Windows 10/11（推荐）、Linux、macOS
- **Python**: 3.10+
- **内存**: 8GB+
- **摄像头**: USB 或内置

### 依赖

```
tensorflow
opencv-python
ultralytics
dlib
pillow
numpy
scikit-learn
pypinyin           # 可选，中文姓名转拼音
```

安装：

```bash
pip install -r FaceCheckPython-master/PeopleFaceCheck/requirements.txt
pip install ultralytics dlib
```

---

## 快速开始

### 1. 采集人脸数据

```bash
python FaceCheckPython-master/PeopleFaceCheck/face_register.py
```

输入姓名 → 自动采集 100 张照片 → 按 ESC 完成。

### 2. 训练 CNN 模型

```bash
cd FaceCheckPython-master/PeopleFaceCheck/CNN_training
python train.py
```

训练完成后模型自动保存到 `model/`，并生成 `num_classes.txt` 记录类别数。

### 3. 运行识别

```bash
python FaceCheckPython-master/PeopleFaceCheck/main/main_yolov8.py
```

---

## 配置说明

### 姓名映射

编辑 `PeopleFaceCheck/name_map.txt`：

```
jinboxiang=靳博翔
zhangsan=张三
human=陌生人
```

左侧是 `training_material/` 下的文件夹名，右侧是显示用的中文名。

### 识别置信度阈值

编辑 `main/main_yolov8.py`：

```python
CONFIDENCE_THRESHOLD = 0.6  # 默认 0.6，低于此值显示"陌生人"
```

### 体型判断阈值

```python
def body_type(w, h):
    ratio = w / h
    if ratio > 0.45:   return "Fat",    (0, 0, 255)
    elif ratio < 0.32: return "Thin",   (255, 0, 255)
    else:              return "Normal",  (0, 255, 0)
```

### 模型类别数

训练时 `num_classes.txt` 自动生成。如需手动指定（例如推理时文件夹数与检查点不一致）：

```
# PeopleFaceCheck/model/num_classes.txt
3
```

---

## 常见问题

### 中文显示为问号或方框

确认系统安装了中文字体（微软雅黑、黑体、宋体），路径在 `C:/Windows/Fonts/`。

### 识别结果总是"陌生人"

- 每人至少 100 张照片
- 各类别数量尽量平衡
- 添加新数据后需重新训练

### Checkpoint restore 报错（shape mismatch）

`dir_num`（类别数）由 `training_material/` 下的文件夹数决定。如果训练后删除了文件夹，推理时图结构与检查点不匹配。解决方法：

1. 确保 `model/num_classes.txt` 中的数值与训练时一致
2. 或重新训练模型

### dlib 安装失败 (Windows)

```bash
pip install cmake
# 下载预编译 whl: https://github.com/z-mahmud22/Dlib_Windows_Python3.x
pip install dlib-xxx.whl
```

---

## 许可证

MIT License

## 作者

邪王真翔

---

## 更新日志

### v2.1 (2025-05-28)
- 修复 checkpoint 类别数不匹配导致 restore 失败的问题
- 训练时自动保存 `num_classes.txt`，推理时优先读取
- 推理时自动为缺失的类别索引生成默认映射
- 帧跳过优化，减少 CNN 推理频率

### v2.0 (2025-05-27)
- 使用 YOLOv8 进行多目标追踪
- 添加体型分析功能
- 优化中文显示（PIL 渲染）
- 添加实时注册功能（R 键）
- 人脸采集工具改用 YOLOv8 检测

### v1.0 (2018)
- 初始版本，基于 dlib + TensorFlow
