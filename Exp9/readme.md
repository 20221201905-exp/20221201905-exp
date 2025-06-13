# 实验报告：基于 LeNet-5 的图像识别算法

**序号**: 24  
**姓名**: 王凯文  
**学号**: 20221201905  

## 一、实验目的

1. 深入理解基于 LeNet-5 卷积神经网络的图像识别算法原理及其在 FashionMNIST 数据集上的应用。
2. 熟练运用 PyTorch 构建 LeNet-5 模型，掌握数据加载、模型训练、评估和可视化的完整流程。
3. 训练模型使得在 FashionMNIST 数据集的测试集上准确率达到 85% 以上。
4. 可视化模型在训练集和测试集上的 loss 曲线和准确度曲线，预测结果与真实结果的对比，以及对应标签的图像。

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

本实验基于 PyTorch 官方提供的 FashionMNIST 数据集，构建并训练一个 LeNet-5 卷积神经网络模型进行图像识别。
- **FashionMNIST**: 包含 60,000 个训练样本和 10,000 个测试样本，28x28 灰度图像，10 类。
实验实现：
1. 训练 LeNet-5 模型，测试集准确率达到 85% 以上。
2. 可视化训练集和测试集的 loss 曲线及准确度曲线。
3. 可视化预测结果与真实结果，并展示对应标签的图像。

## 四、实验步骤

### 1. 环境搭建

在 Windows 10 系统中，确保已安装 Python 3.9。通过命令行工具执行以下命令安装所需库：

```bash
pip install torch torchvision matplotlib numpy
```

### 2. 数据预处理

此阶段完成 FashionMNIST 数据集的加载和预处理。

**代码**:  
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```

**数据预处理分析**:  
- **数据集规模**: 训练集 60,000 个样本，测试集 10,000 个样本。
- **特征与目标**: 28x28 像素值（单通道）作为特征，10 个类别（0-9）作为目标变量。
- **预处理**: 使用 ToTensor() 转换为张量，Normalize() 标准化（均值 0.5，标准差 0.5）。

### 3. 探索性数据分析 (EDA)

通过可视化少量图像，了解数据分布。

**代码** (示例):  
```python
images, labels = next(iter(train_loader))
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i].numpy().squeeze(), cmap='gray')
    plt.title(f'Label: {labels[i].item()}')
    plt.axis('off')
plt.show()
```

**EDA 结果分析**:  
- **图像分布**: 前 5 张图像显示类别如 T-shirt 和 Trouser，灰度值分布反映笔迹差异。
- **数据特性**: 28x28 像素的高维数据适合卷积神经网络分类。

### 4. 模型训练

构建 LeNet-5 卷积神经网络并进行训练。

**代码**:  
```python
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 20
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
- **网络结构**: LeNet-5，包含两个卷积层 (Conv2d) 和三个全连接层 (FC)，适配 28x28 单通道输入。
- **优化**: 使用 SGD (lr=0.01, momentum=0.9)，20 个 epoch，batch_size=64。
- **性能**: 测试集准确率从 79.12% 升至 90.44%。

### 5. 模型评估

评估模型在测试集上的性能。

**输出结果** (基于运行结果):  
```
Epoch [1/20], Train Loss: 0.7777, Train Acc: 70.54%, Test Loss: 0.5340, Test Acc: 79.12%
Epoch [2/20], Train Loss: 0.4625, Train Acc: 82.71%, Test Loss: 0.4413, Test Acc: 83.44%
Epoch [3/20], Train Loss: 0.3955, Train Acc: 85.37%, Test Loss: 0.4062, Test Acc: 85.22%
Epoch [4/20], Train Loss: 0.3595, Train Acc: 86.72%, Test Loss: 0.3894, Test Acc: 85.95%
Epoch [5/20], Train Loss: 0.3272, Train Acc: 87.80%, Test Loss: 0.3744, Test Acc: 87.33%
Epoch [6/20], Train Loss: 0.3026, Train Acc: 88.88%, Test Loss: 0.3639, Test Acc: 87.39%
Epoch [7/20], Train Loss: 0.2866, Train Acc: 89.33%, Test Loss: 0.3409, Test Acc: 88.52%
Epoch [8/20], Train Loss: 0.2740, Train Acc: 89.83%, Test Loss: 0.3280, Test Acc: 88.68%
Epoch [9/20], Train Loss: 0.2569, Train Acc: 90.19%, Test Loss: 0.3297, Test Acc: 89.14%
Epoch [10/20], Train Loss: 0.2418, Train Acc: 90.67%, Test Loss: 0.3208, Test Acc: 89.30%
Epoch [11/20], Train Loss: 0.2337, Train Acc: 91.06%, Test Loss: 0.3098, Test Acc: 89.37%
Epoch [12/20], Train Loss: 0.2238, Train Acc: 91.31%, Test Loss: 0.2983, Test Acc: 89.51%
Epoch [13/20], Train Loss: 0.2138, Train Acc: 91.67%, Test Loss: 0.2974, Test Acc: 90.07%
Epoch [14/20], Train Loss: 0.2054, Train Acc: 92.06%, Test Loss: 0.2888, Test Acc: 90.12%
Epoch [15/20], Train Loss: 0.1975, Train Acc: 92.40%, Test Loss: 0.2792, Test Acc: 90.26%
Epoch [16/20], Train Loss: 0.1907, Train Acc: 92.62%, Test Loss: 0.2762, Test Acc: 90.39%
Epoch [17/20], Train Loss: 0.1846, Train Acc: 92.94%, Test Loss: 0.2656, Test Acc: 90.42%
Epoch [18/20], Train Loss: 0.1786, Train Acc: 93.19%, Test Loss: 0.2636, Test Acc: 90.71%
Epoch [19/20], Train Loss: 0.1724, Train Acc: 93.39%, Test Loss: 0.2736, Test Acc: 90.14%
Epoch [20/20], Train Loss: 0.1636, Train Acc: 93.96%, Test Loss: 0.2736, Test Acc: 90.44%
```

**评估分析**:  
- **测试集准确率**: 从 79.12% 升至 90.44%（第 20 个 epoch），满足 ≥ 85% 要求。训练集准确率 (93.96%) 高于测试集，存在轻微过拟合。
- **损失趋势**: 训练集 loss 从 0.7777 降至 0.1636，测试集 loss 从 0.5340 降至 0.2736，收敛良好。
- **性能**: LeNet-5 在 FashionMNIST 上表现出色，准确率显著高于全连接网络。

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
    plt.suptitle('FashionMNIST Predictions')
    plt.tight_layout()
    plt.show()
```

**可视化分析**:  
- **Loss 曲线**: 训练集 loss 从 0.7777 降至 0.1636，测试集 loss 稳定在 0.27 左右，收敛良好。
- **准确度曲线**: 训练集准确率从 70.54% 升至 93.96%，测试集从 79.12% 升至 90.44%，趋势一致。
- **预测结果**: 10 张测试图像中，预测标签与真实标签高度一致，误分类较少。

## 五、实验结果与分析

本次实验成功实现了基于 LeNet-5 的 FashionMNIST 图像识别。

### 实验结果总结:  
- **模型性能**: LeNet-5 在测试集上准确率 90.44%（第 20 个 epoch），满足 ≥ 85% 要求。
- **损失与准确度**: 训练集和测试集 loss 收敛，准确度稳步提升。
- **可视化验证**: 曲线显示模型收敛，预测结果与图像一致性高。

### 局限性与展望:  
- **过拟合风险**: 训练集准确率 (93.96%) 高于测试集 (90.44%)，未来可加入 Dropout。
- **网络结构**: LeNet-5 较为简单，未来可尝试更深层网络。
- **超参数**: 未优化学习率，未来可尝试调度。

### 结论:  
本次实验通过 LeNet-5 实现了 FashionMNIST 图像识别，测试集准确率 90.44%，满足要求。卷积神经网络显著优于全连接网络，掌握了 CNN 构建与训练方法。

**未来展望**:  
- 加入 Dropout 或 L2 正则化，减少过拟合。
- 尝试更深层 CNN（如 AlexNet）提升性能。
- 优化超参数，探索学习率调度。

本次实验为后续深度学习学习奠定了基础。