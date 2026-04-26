import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

# 路径设置
BASE_DIR = "./chest_xray"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ===================== 1. 数据预处理与加载 =====================
# 训练集增强（防过拟合）
train_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 测试/验证集（无增强）
test_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载原始训练集（用于重新划分）
full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

# 避坑：从train按8:2划分训练/验证
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# 数据加载器
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 数据集统计
print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
print(f"类别: {full_train_dataset.class_to_idx}")

# ===================== 2. 搭建简易CNN模型 =====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 18 * 18, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 18 * 18)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

model = SimpleCNN().to(device)

# ===================== 3. 训练配置 =====================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 15

# 记录
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# ===================== 4. 训练与验证 =====================
print("开始训练...")
start_time = time.time()

for epoch in range(EPOCHS):
    # 训练
    model.train()
    train_loss = 0.0
    train_preds, train_labels = [], []
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = (outputs > 0.5).float()
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    train_acc = accuracy_score(train_labels, train_preds)
    train_losses.append(train_loss/len(train_loader))
    train_accs.append(train_acc)

    # 验证
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (outputs > 0.5).float()
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_acc = accuracy_score(val_labels, val_preds)
    val_losses.append(val_loss/len(val_loader))
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.4f}")

train_time = round(time.time() - start_time, 2)
print(f"训练完成，总耗时: {train_time}s")

# ===================== 5. 测试集评估 =====================
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = (outputs > 0.5).cpu().numpy().astype(int).flatten()
        test_preds.extend(preds)
        test_labels.extend(labels.numpy())

# 四项指标
acc = accuracy_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)

print("\n===== 测试集结果 =====")
print(f"准确率(Accuracy): {acc:.4f}")
print(f"精确率(Precision): {precision:.4f}")
print(f"召回率(Recall): {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# ===================== 6. 绘图保存 =====================
# 损失曲线
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve")
plt.legend()

# 准确率曲线
plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.title("Accuracy Curve")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/loss_acc_curve.png")
plt.close()

# 混淆矩阵
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Pneumonia"], yticklabels=["Normal","Pneumonia"])
plt.title("Confusion Matrix")
plt.xlabel("Pred")
plt.ylabel("True")
plt.savefig(f"{FIG_DIR}/confusion_matrix.png")
plt.close()