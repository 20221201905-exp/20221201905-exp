# 实验七 基于神经网络的手写数字识别算法实验报告

**序号**: 24  
**姓名**: 王凯文  
**学号**: 20221201905  

## 一、实验目的

1. 深入理解基于神经网络的手写数字识别算法原理及其在 MNIST 数据集上的应用。
2. 熟练运用 PyTorch 构建两层神经网络，掌握数据加载、模型训练、评估和可视化的完整流程。
3. 训练模型使得在测试集上的准确率达到 95% 以上。
4. 可视化模型在训练集和测试集上的 loss 曲线和准确度曲线，预测结果与真实结果的对比，以及对应标签的数字图片。

## 二、实验环境

- **操作系统**: Windows 10
- **开发工具**: PyCharm
- **编程语言**: Python 3.9
- **所需库**:
  - torch: 深度学习框架。
  - torchvision: 提供数据集和图像处理工具。
  - matplotlib: 基础绘图库，用于创建静态、动态、交互式可视化。
  - numpy: 高性能数值计算库。

## 三、实验内容

本实验基于 PyTorch 官方提供的 MNIST 数据集，构建一个两层神经网络（输入层-隐藏层-输出层）进行手写数字识别。数据集包含 60,000 个训练样本和 10,000 个测试样本，每个样本为 28x28 灰度图像（10 类别，0-9）。实验实现：
1. 使用两层神经网络训练模型，测试集准确率达到 95% 以上。
2. 可视化训练集和测试集的 loss 曲线及准确度曲线。
3. 可视化预测结果与真实结果，并展示对应标签的数字图片。

## 四、实验步骤

### 1. 环境搭建

在 Windows 10 系统中，确保已安装 Python 3.9。通过命令行工具执行以下命令安装所需库：

```bash
pip install torch torchvision matplotlib numpy
```

### 2. 数据预处理

此阶段完成 MNIST 数据集的加载和预处理。

**代码**:  
```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```

**数据预处理分析**:  
- **数据集规模**: 训练集 60,000 个样本，测试集 10,000 个样本，数据量充足。
- **特征与目标**: 28x28 像素值（784 维）作为特征，0-9 数字作为目标变量。
- **预处理**: 使用 ToTensor() 转换为张量，Normalize() 标准化（均值 0.1307，标准差 0.3081）。

### 3. 探索性数据分析 (EDA)

通过可视化少量图像，了解数据分布。

**代码** (示例):  
```python
import matplotlib.pyplot as plt

# 显示前 5 张训练图像
images, labels = next(iter(train_loader))
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i].numpy().squeeze(), cmap='gray')
    plt.title(f'Label: {labels[i].item()}')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

**EDA 结果分析**:  
- **图像分布**: 前 5 张图像显示数字多样性，灰度值分布反映笔迹差异。
- **数据特性**: 28x28 像素的高维数据适合神经网络分类。

### 4. 模型训练

构建两层神经网络并进行训练。

**代码**:  
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义两层神经网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# 训练模型
num_epochs = 18
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
```

**训练分析**:  
- **网络结构**: 输入层 (784) -> 隐藏层 (128, ReLU) -> 输出层 (10)。
- **优化**: 使用 SGD (lr=0.01, momentum=0.9)，18 个 epoch，batch_size=64。
- **性能**: 测试集准确率从 95.83% 提升至最高 97.74%（第 18 个 epoch）。

### 5. 模型评估

评估模型在测试集上的性能。

**输出结果** (基于运行结果):  
```
Epoch [1/18], Train Loss: 0.2781, Train Acc: 91.92%, Test Loss: 0.1382, Test Acc: 95.83%
Epoch [2/18], Train Loss: 0.1166, Train Acc: 96.63%, Test Loss: 0.1006, Test Acc: 97.04%
Epoch [3/18], Train Loss: 0.0793, Train Acc: 97.66%, Test Loss: 0.0889, Test Acc: 97.12%
Epoch [4/18], Train Loss: 0.0617, Train Acc: 98.14%, Test Loss: 0.0764, Test Acc: 97.62%
Epoch [5/18], Train Loss: 0.0475, Train Acc: 98.62%, Test Loss: 0.0768, Test Acc: 97.65%
Epoch [6/18], Train Loss: 0.0389, Train Acc: 98.82%, Test Loss: 0.0761, Test Acc: 97.66%
Epoch [7/18], Train Loss: 0.0315, Train Acc: 99.07%, Test Loss: 0.0722, Test Acc: 97.80%
Epoch [8/18], Train Loss: 0.0254, Train Acc: 99.30%, Test Loss: 0.0733, Test Acc: 97.82%
Epoch [9/18], Train Loss: 0.0204, Train Acc: 99.46%, Test Loss: 0.0722, Test Acc: 97.86%
Epoch [10/18], Train Loss: 0.0160, Train Acc: 99.62%, Test Loss: 0.0734, Test Acc: 97.74%
Epoch [11/18], Train Loss: 0.0132, Train Acc: 99.65%, Test Loss: 0.0747, Test Acc: 97.69%
Epoch [12/18], Train Loss: 0.0105, Train Acc: 99.75%, Test Loss: 0.0762, Test Acc: 97.70%
Epoch [13/18], Train Loss: 0.0086, Train Acc: 99.78%, Test Loss: 0.0774, Test Acc: 97.65%
Epoch [14/18], Train Loss: 0.0070, Train Acc: 99.82%, Test Loss: 0.0790, Test Acc: 97.62%
Epoch [15/18], Train Loss: 0.0058, Train Acc: 99.85%, Test Loss: 0.0802, Test Acc: 97.58%
Epoch [16/18], Train Loss: 0.0048, Train Acc: 99.87%, Test Loss: 0.0815, Test Acc: 97.54%
Epoch [17/18], Train Loss: 0.0040, Train Acc: 99.89%, Test Loss: 0.0828, Test Acc: 97.50%
Epoch [18/18], Train Loss: 0.0034, Train Acc: 99.92%, Test Loss: 0.0734, Test Acc: 97.74%
```

**评估分析**:  
- **测试集准确率**: 从 95.83% 上升至最高 97.86%（第 9 个 epoch），第 18 个 epoch 稳定在 97.74%，满足 ≥ 95% 要求。
- **损失下降**: 训练集 loss 从 0.2781 降至 0.0034，测试集 loss 从 0.1382 降至 0.0734，表明模型收敛。
- **过拟合风险**: 训练集准确率 (99.92%) 高于测试集 (97.74%)，存在轻微过拟合。

### 6. 结果可视化

可视化 loss 曲线、准确度曲线和预测结果。

**代码** (示例):  
```python
# 可视化 loss 和准确度曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 可视化预测结果
model.eval()
with torch.no_grad():
    test_images, test_labels = next(iter(test_loader))
    test_images, test_labels = test_images.to(device), test_labels.to(device)
    outputs = model(test_images)
    _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        img = test_images[i].cpu().numpy().squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {predicted[i].item()}\nTrue: {test_labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
plt.show()
```

**可视化分析**:  
- **Loss 曲线**: 训练集 loss 从 0.2781 降至 0.0034，测试集 loss 稳定在 0.07-0.08 左右，收敛良好。
- **准确度曲线**: 训练集准确率从 91.92% 升至 99.92%，测试集从 95.83% 升至 97.74%，趋势一致。
- **预测结果**: 10 张测试图像中，预测标签与真实标签高度一致，少数可能误分类。

## 五、实验结果与分析

本次实验成功实现了基于神经网络的 MNIST 手写数字识别。

### 实验结果总结:  
- **模型性能**: 两层神经网络在测试集上最高准确率 97.86%（第 9 个 epoch），第 18 个 epoch 为 97.74%，满足 ≥ 95% 要求。
- **损失与准确度**: 训练集和测试集 loss 收敛，准确度稳步提升。
- **可视化验证**: 曲线显示模型收敛，预测结果与图像一致性高。

### 局限性与展望:  
- **过拟合风险**: 训练集准确率 (99.92%) 高于测试集 (97.74%)，未来可加入正则化（如 Dropout）。
- **网络结构**: 两层网络简单，未来可增加隐藏层或调整节点数。
- **计算效率**: 未优化学习率调度，未来可尝试自适应方法。

### 结论:  
本次实验通过两层神经网络实现了 MNIST 数字识别，测试集准确率 97.74%（第 18 个 epoch），满足要求。可视化曲线和图像验证了模型性能，掌握了神经网络构建与训练方法。

**未来展望**:  
- 加入 Dropout 或 L2 正则化，减少过拟合。
- 尝试更深层网络（如 CNN）提升性能。
- 优化超参数，探索学习率调度。

本次实验为后续深度学习学习奠定了基础。