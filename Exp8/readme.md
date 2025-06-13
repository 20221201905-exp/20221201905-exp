# 实验八 基于神经网络的图像识别算法实验报告

**序号**: 24  
**姓名**: 王凯文  
**学号**: 20221201905  

## 一、实验目的

1. 深入理解基于神经网络的图像识别算法原理及其在 FashionMNIST 和 CIFAR-10 数据集上的应用。
2. 熟练运用 PyTorch 构建三层神经网络，掌握数据加载、模型训练、评估和可视化的完整流程。
3. 训练模型使得在 FashionMNIST 数据集的测试集上准确率达到 85% 以上。
4. 训练模型观察神经网络在 CIFAR-10 测试集上的表现。
5. 可视化模型在训练集和测试集上的 loss 曲线和准确度曲线，预测结果与真实结果的对比，以及对应标签的图像。

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

本实验基于 PyTorch 官方提供的 FashionMNIST 和 CIFAR-10 数据集，构建一个三层神经网络（输入层-隐藏层1-隐藏层2-输出层）进行图像识别。
- **FashionMNIST**: 包含 60,000 个训练样本和 10,000 个测试样本，28x28 灰度图像，10 类。
- **CIFAR-10**: 包含 50,000 个训练样本和 10,000 个测试样本，32x32 RGB 图像，10 类。
实验实现：
1. 训练 FashionMNIST 模型，测试集准确率达到 85% 以上。
2. 训练 CIFAR-10 模型，观察其测试集表现。
3. 可视化训练集和测试集的 loss 曲线及准确度曲线。
4. 可视化预测结果与真实结果，并展示对应标签的图像。

## 四、实验步骤

### 1. 环境搭建

在 Windows 10 系统中，确保已安装 Python 3.9。通过命令行工具执行以下命令安装所需库：

```bash
pip install torch torchvision matplotlib numpy
```

### 2. 数据预处理

此阶段完成 FashionMNIST 和 CIFAR-10 数据集的加载和预处理。

**代码**:  
```python
# FashionMNIST 使用单通道归一化
fashion_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# CIFAR-10 使用三通道归一化
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

fashion_train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=fashion_transform)
fashion_test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=fashion_transform)
fashion_train_loader = DataLoader(dataset=fashion_train_dataset, batch_size=64, shuffle=True)
fashion_test_loader = DataLoader(dataset=fashion_test_dataset, batch_size=64, shuffle=False)

cifar_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
cifar_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
cifar_train_loader = DataLoader(dataset=cifar_train_dataset, batch_size=64, shuffle=True)
cifar_test_loader = DataLoader(dataset=cifar_test_dataset, batch_size=64, shuffle=False)
```

**数据预处理分析**:  
- **FashionMNIST**: 28x28 单通道，标准化均值 0.5，标准差 0.5。
- **CIFAR-10**: 32x32 三通道，标准化均值 (0.5, 0.5, 0.5)，标准差 (0.5, 0.5, 0.5)。
- **数据规模**: FashionMNIST 60,000 训练，10,000 测试；CIFAR-10 50,000 训练，10,000 测试。

### 3. 探索性数据分析 (EDA)

通过可视化少量图像，了解数据分布。

**代码** (示例):  
```python
# 显示 FashionMNIST 前 5 张图像
fashion_images, fashion_labels = next(iter(fashion_train_loader))
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(fashion_images[i].numpy().squeeze(), cmap='gray')
    plt.title(f'Label: {fashion_labels[i].item()}')
    plt.axis('off')
plt.show()

# 显示 CIFAR-10 前 5 张图像
cifar_images, cifar_labels = next(iter(cifar_train_loader))
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(cifar_images[i].numpy().transpose((1, 2, 0)))
    plt.title(f'Label: {cifar_labels[i].item()}')
    plt.axis('off')
plt.show()
```

**EDA 结果分析**:  
- **FashionMNIST**: 灰度图像，类别如 T-shirt 和 Trouser 分布清晰。
- **CIFAR-10**: RGB 图像，类别如 cat 和 dog 更复杂，特征丰富。

### 4. 模型训练

构建三层神经网络并进行训练。

**代码**:  
```python
# 定义三层神经网络
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fashion_model = NeuralNetwork(input_size=28*28, hidden1_size=256, hidden2_size=128, num_classes=10).to(device)
cifar_model = NeuralNetwork(input_size=32*32*3, hidden1_size=512, hidden2_size=256, num_classes=10).to(device)
fashion_criterion = nn.CrossEntropyLoss()
cifar_criterion = nn.CrossEntropyLoss()
fashion_optimizer = optim.SGD(fashion_model.parameters(), lr=0.01, momentum=0.9)
cifar_optimizer = optim.SGD(cifar_model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
num_epochs = 20
fashion_train_losses, fashion_train_accs = [], []
fashion_test_losses, fashion_test_accs = [], []
cifar_train_losses, cifar_train_accs = [], []
cifar_test_losses, cifar_test_accs = [], []

for epoch in range(num_epochs):
    fashion_train_loss, fashion_train_acc = train(fashion_model, fashion_train_loader, fashion_criterion, fashion_optimizer, epoch)
    fashion_test_loss, fashion_test_acc = evaluate(fashion_model, fashion_test_loader, fashion_criterion)
    fashion_train_losses.append(fashion_train_loss)
    fashion_train_accs.append(fashion_train_acc)
    fashion_test_losses.append(fashion_test_loss)
    fashion_test_accs.append(fashion_test_acc)
    print(f'[FashionMNIST] Epoch [{epoch+1}/{num_epochs}], Train Loss: {fashion_train_loss:.4f}, Train Acc: {fashion_train_acc:.2f}%, '
          f'Test Loss: {fashion_test_loss:.4f}, Test Acc: {fashion_test_acc:.2f}%')

    cifar_train_loss, cifar_train_acc = train(cifar_model, cifar_train_loader, cifar_criterion, cifar_optimizer, epoch)
    cifar_test_loss, cifar_test_acc = evaluate(cifar_model, cifar_test_loader, cifar_criterion)
    cifar_train_losses.append(cifar_train_loss)
    cifar_train_accs.append(cifar_train_acc)
    cifar_test_losses.append(cifar_test_loss)
    cifar_test_accs.append(cifar_test_acc)
    print(f'[CIFAR-10] Epoch [{epoch+1}/{num_epochs}], Train Loss: {cifar_train_loss:.4f}, Train Acc: {cifar_train_acc:.2f}%, '
          f'Test Loss: {cifar_test_loss:.4f}, Test Acc: {cifar_test_acc:.2f}%')
```

**训练分析**:  
- **网络结构**: FashionMNIST (784 -> 256 -> 128 -> 10)，CIFAR-10 (3072 -> 512 -> 256 -> 10)。
- **优化**: SGD (lr=0.01, momentum=0.9)，20 个 epoch，batch_size=64。
- **性能**: FashionMNIST 测试集达 89.46%，CIFAR-10 达 53.84%。

### 5. 模型评估

评估模型在测试集上的性能。

**输出结果** (基于运行结果):  
```
[FashionMNIST] Epoch [1/20], Train Loss: 0.5751, Train Acc: 79.61%, Test Loss: 0.4668, Test Acc: 82.65%
[FashionMNIST] Epoch [2/20], Train Loss: 0.3966, Train Acc: 85.69%, Test Loss: 0.4148, Test Acc: 84.25%
[FashionMNIST] Epoch [3/20], Train Loss: 0.3496, Train Acc: 87.03%, Test Loss: 0.3791, Test Acc: 86.20%
[FashionMNIST] Epoch [4/20], Train Loss: 0.3226, Train Acc: 87.97%, Test Loss: 0.3850, Test Acc: 85.34%
[FashionMNIST] Epoch [5/20], Train Loss: 0.3051, Train Acc: 88.62%, Test Loss: 0.3733, Test Acc: 86.66%
[FashionMNIST] Epoch [6/20], Train Loss: 0.2897, Train Acc: 89.27%, Test Loss: 0.3440, Test Acc: 87.60%
[FashionMNIST] Epoch [7/20], Train Loss: 0.2748, Train Acc: 89.77%, Test Loss: 0.3617, Test Acc: 87.02%
[FashionMNIST] Epoch [8/20], Train Loss: 0.2580, Train Acc: 90.32%, Test Loss: 0.3320, Test Acc: 88.16%
[FashionMNIST] Epoch [9/20], Train Loss: 0.2508, Train Acc: 90.63%, Test Loss: 0.3369, Test Acc: 88.14%
[FashionMNIST] Epoch [10/20], Train Loss: 0.2398, Train Acc: 91.02%, Test Loss: 0.3322, Test Acc: 88.41%
[FashionMNIST] Epoch [11/20], Train Loss: 0.2298, Train Acc: 91.30%, Test Loss: 0.3426, Test Acc: 88.35%
[FashionMNIST] Epoch [12/20], Train Loss: 0.2219, Train Acc: 91.55%, Test Loss: 0.3347, Test Acc: 88.35%
[FashionMNIST] Epoch [13/20], Train Loss: 0.2119, Train Acc: 92.08%, Test Loss: 0.3315, Test Acc: 88.63%
[FashionMNIST] Epoch [14/20], Train Loss: 0.2059, Train Acc: 92.40%, Test Loss: 0.3232, Test Acc: 88.92%
[FashionMNIST] Epoch [15/20], Train Loss: 0.1953, Train Acc: 92.72%, Test Loss: 0.3328, Test Acc: 88.25%
[FashionMNIST] Epoch [16/20], Train Loss: 0.1909, Train Acc: 92.82%, Test Loss: 0.3321, Test Acc: 88.19%
[FashionMNIST] Epoch [17/20], Train Loss: 0.1821, Train Acc: 93.07%, Test Loss: 0.3367, Test Acc: 88.77%
[FashionMNIST] Epoch [18/20], Train Loss: 0.1764, Train Acc: 93.42%, Test Loss: 0.3376, Test Acc: 87.81%
[FashionMNIST] Epoch [19/20], Train Loss: 0.1697, Train Acc: 93.68%, Test Loss: 0.3549, Test Acc: 88.48%
[FashionMNIST] Epoch [20/20], Train Loss: 0.1644, Train Acc: 93.92%, Test Loss: 0.3468, Test Acc: 89.46%

[CIFAR-10] Epoch [1/20], Train Loss: 1.6947, Train Acc: 39.65%, Test Loss: 1.5133, Test Acc: 46.63%
[CIFAR-10] Epoch [2/20], Train Loss: 1.4359, Train Acc: 49.42%, Test Loss: 1.4218, Test Acc: 49.49%
[CIFAR-10] Epoch [3/20], Train Loss: 1.3219, Train Acc: 53.42%, Test Loss: 1.3669, Test Acc: 51.26%
[CIFAR-10] Epoch [4/20], Train Loss: 1.2274, Train Acc: 56.74%, Test Loss: 1.3475, Test Acc: 52.54%
[CIFAR-10] Epoch [5/20], Train Loss: 1.1564, Train Acc: 59.50%, Test Loss: 1.3214, Test Acc: 53.41%
[CIFAR-10] Epoch [6/20], Train Loss: 1.0864, Train Acc: 62.42%, Test Loss: 1.3431, Test Acc: 53.92%
[CIFAR-10] Epoch [7/20], Train Loss: 1.0166, Train Acc: 64.62%, Test Loss: 1.3090, Test Acc: 54.68%
[CIFAR-10] Epoch [8/20], Train Loss: 0.9416, Train Acc: 66.98%, Test Loss: 1.3947, Test Acc: 53.43%
[CIFAR-10] Epoch [9/20], Train Loss: 0.8735, Train Acc: 70.10%, Test Loss: 1.4671, Test Acc: 52.54%
[CIFAR-10] Epoch [10/20], Train Loss: 0.8097, Train Acc: 71.25%, Test Loss: 1.4671, Test Acc: 53.43%
[CIFAR-10] Epoch [11/20], Train Loss: 0.7529, Train Acc: 73.69%, Test Loss: 1.4915, Test Acc: 54.66%
[CIFAR-10] Epoch [12/20], Train Loss: 0.6894, Train Acc: 76.31%, Test Loss: 1.5075, Test Acc: 54.59%
[CIFAR-10] Epoch [13/20], Train Loss: 0.6431, Train Acc: 77.20%, Test Loss: 1.6087, Test Acc: 54.47%
[CIFAR-10] Epoch [14/20], Train Loss: 0.5949, Train Acc: 79.35%, Test Loss: 1.6357, Test Acc: 54.69%
[CIFAR-10] Epoch [15/20], Train Loss: 0.5484, Train Acc: 80.61%, Test Loss: 1.7373, Test Acc: 54.37%
[CIFAR-10] Epoch [16/20], Train Loss: 0.5011, Train Acc: 82.06%, Test Loss: 1.7177, Test Acc: 54.87%
[CIFAR-10] Epoch [17/20], Train Loss: 0.4585, Train Acc: 84.51%, Test Loss: 1.8346, Test Acc: 54.45%
[CIFAR-10] Epoch [18/20], Train Loss: 0.4188, Train Acc: 85.73%, Test Loss: 1.9650, Test Acc: 53.71%
[CIFAR-10] Epoch [19/20], Train Loss: 0.3794, Train Acc: 86.69%, Test Loss: 2.0771, Test Acc: 53.73%
[CIFAR-10] Epoch [20/20], Train Loss: 0.3560, Train Acc: 87.46%, Test Loss: 2.1942, Test Acc: 53.84%
```

**评估分析**:  
- **FashionMNIST**: 测试集准确率从 82.65% 升至 89.46%（第 20 个 epoch），满足 ≥ 85% 要求。训练集准确率 (93.92%) 高于测试集，存在轻微过拟合。
- **CIFAR-10**: 测试集准确率从 46.63% 升至 53.84%（第 20 个 epoch），反映复杂数据集和简单网络的局限性。训练集准确率 (87.46%) 高于测试集，过拟合更明显。
- **损失趋势**: FashionMNIST loss 降至 0.3468，CIFAR-10 loss 升至 2.1942，表明 CIFAR-10 模型可能未充分收敛。

### 6. 结果可视化

可视化 loss 曲线、准确度曲线和预测结果。

**代码** (示例):  
```python
# 可视化 loss 和准确度曲线
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(fashion_train_losses, label='Train Loss')
plt.plot(fashion_test_losses, label='Test Loss')
plt.title('FashionMNIST Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(fashion_train_accs, label='Train Accuracy')
plt.plot(fashion_test_accs, label='Test Accuracy')
plt.title('FashionMNIST Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(cifar_train_losses, label='Train Loss')
plt.plot(cifar_test_losses, label='Test Loss')
plt.title('CIFAR-10 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(cifar_train_accs, label='Train Accuracy')
plt.plot(cifar_test_accs, label='Test Accuracy')
plt.title('CIFAR-10 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 可视化预测结果
fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

visualize_predictions(fashion_model, fashion_test_loader, 'FashionMNIST', fashion_classes)
visualize_predictions(cifar_model, cifar_test_loader, 'CIFAR-10', cifar_classes)
```

**可视化分析**:  
- **FashionMNIST**: Loss 从 0.5751 降至 0.3468，准确度从 79.61% 升至 89.46%，收敛良好。
- **CIFAR-10**: Loss 从 1.6947 降至 0.3560 后上升至 2.1942，准确度从 39.65% 升至 53.84%，收敛不佳。
- **预测结果**: FashionMNIST 预测准确率高，CIFAR-10 误分类较多。

## 五、实验结果与分析

本次实验实现了基于神经网络的 FashionMNIST 和 CIFAR-10 图像识别。

### 实验结果总结:  
- **FashionMNIST**: 测试集准确率 89.46%（第 20 个 epoch），满足 ≥ 85% 要求。
- **CIFAR-10**: 测试集准确率 53.84%（第 20 个 epoch），反映网络简单性限制。
- **可视化验证**: 曲线显示 FashionMNIST 收敛良好，CIFAR-10 收敛较差。

### 局限性与展望:  
- **CIFAR-10 性能**: 53.84% 较低，未来可使用 CNN。
- **过拟合**: 训练集准确率高于测试集，未来可加入 Dropout。
- **超参数**: 未优化学习率，未来可尝试调度。

### 结论:  
本次实验通过三层神经网络实现了 FashionMNIST 识别（89.46%），满足要求；CIFAR-10 达到 53.84%，验证了数据复杂性影响。掌握了神经网络构建与训练方法。

**未来展望**:  
- 使用 CNN 提升 CIFAR-10 性能。
- 加入 Dropout 或 L2 正则化。
- 优化超参数，探索学习率调度。

本次实验为后续深度学习学习奠定了基础。