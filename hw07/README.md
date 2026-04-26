# HW07：胸部X光肺炎影像二分类实战
## 目录结构
hw07/
├── train.py          # 训练+评估主代码
├── requirements.txt  # 依赖
├── report.md         # 实验报告
├── debug_notes.md    # 调试记录
├── figures/          # 训练曲线、混淆矩阵
└── chest_xray/       # 数据集目录（需自行下载）

## 数据集获取
1. 下载地址：https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. 解压后将`chest_xray`文件夹放在hw07根目录

## 运行命令
# 安装依赖
pip install -r requirements.txt
# 运行训练
python train.py

## 核心说明
1. 验证集：从训练集按8:2重新划分，未使用原始16张val
2. 模型：3层卷积+全连接简易CNN
3. 评估指标：Accuracy、Precision、Recall、F1
4. 输出：训练曲线、混淆矩阵保存至figures/

## 测试集结果
- 准确率: ~0.92
- 精确率: ~0.94
- 召回率: ~0.95
- F1分数: ~0.94