import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保可复现性
torch.manual_seed(42)

# 数据预处理
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

# 加载 FashionMNIST 数据集
fashion_train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=fashion_transform)
fashion_test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=fashion_transform)
fashion_train_loader = DataLoader(dataset=fashion_train_dataset, batch_size=64, shuffle=True)
fashion_test_loader = DataLoader(dataset=fashion_test_dataset, batch_size=64, shuffle=False)

# 加载 CIFAR-10 数据集
cifar_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
cifar_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
cifar_train_loader = DataLoader(dataset=cifar_train_dataset, batch_size=64, shuffle=True)
cifar_test_loader = DataLoader(dataset=cifar_test_dataset, batch_size=64, shuffle=False)

# 定义三层神经网络
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden1_size),  # 输入层到隐藏层1
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),  # 隐藏层1到隐藏层2
            nn.ReLU(),
            nn.Linear(hidden2_size, num_classes)   # 隐藏层2到输出层
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FashionMNIST 模型 (28x28=784 输入，10 类)
fashion_model = NeuralNetwork(input_size=28*28, hidden1_size=256, hidden2_size=128, num_classes=10).to(device)
fashion_criterion = nn.CrossEntropyLoss()
fashion_optimizer = optim.SGD(fashion_model.parameters(), lr=0.01, momentum=0.9)

# CIFAR-10 模型 (32x32x3=3072 输入，10 类)
cifar_model = NeuralNetwork(input_size=32*32*3, hidden1_size=512, hidden2_size=256, num_classes=10).to(device)
cifar_criterion = nn.CrossEntropyLoss()
cifar_optimizer = optim.SGD(cifar_model.parameters(), lr=0.01, momentum=0.9)

# 训练和评估函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
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

def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

# 训练 FashionMNIST 模型
num_epochs = 20
fashion_train_losses, fashion_train_accs = [], []
fashion_test_losses, fashion_test_accs = [], []

for epoch in range(num_epochs):
    fashion_train_loss, fashion_train_acc = train(fashion_model, fashion_train_loader, fashion_criterion, fashion_optimizer, epoch)
    fashion_test_loss, fashion_test_acc = evaluate(fashion_model, fashion_test_loader, fashion_criterion)
    fashion_train_losses.append(fashion_train_loss)
    fashion_train_accs.append(fashion_train_acc)
    fashion_test_losses.append(fashion_test_loss)
    fashion_test_accs.append(fashion_test_acc)
    print(f'[FashionMNIST] Epoch [{epoch+1}/{num_epochs}], Train Loss: {fashion_train_loss:.4f}, Train Acc: {fashion_train_acc:.2f}%, '
          f'Test Loss: {fashion_test_loss:.4f}, Test Acc: {fashion_test_acc:.2f}%')

# 训练 CIFAR-10 模型
cifar_train_losses, cifar_train_accs = [], []
cifar_test_losses, cifar_test_accs = [], []

for epoch in range(num_epochs):
    cifar_train_loss, cifar_train_acc = train(cifar_model, cifar_train_loader, cifar_criterion, cifar_optimizer, epoch)
    cifar_test_loss, cifar_test_acc = evaluate(cifar_model, cifar_test_loader, cifar_criterion)
    cifar_train_losses.append(cifar_train_loss)
    cifar_train_accs.append(cifar_train_acc)
    cifar_test_losses.append(cifar_test_loss)
    cifar_test_accs.append(cifar_test_acc)
    print(f'[CIFAR-10] Epoch [{epoch+1}/{num_epochs}], Train Loss: {cifar_train_loss:.4f}, Train Acc: {cifar_train_acc:.2f}%, '
          f'Test Loss: {cifar_test_loss:.4f}, Test Acc: {cifar_test_acc:.2f}%')

# 可视化 loss 和准确度曲线
plt.figure(figsize=(12, 8))

# FashionMNIST
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

# CIFAR-10
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
def visualize_predictions(model, test_loader, dataset_name, class_names):
    model.eval()
    with torch.no_grad():
        test_images, test_labels = next(iter(test_loader))
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        outputs = model(test_images)
        _, predicted = torch.max(outputs, 1)

        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i in range(10):
            ax = axes[i // 5, i % 5]
            img = test_images[i].cpu().numpy().transpose((1, 2, 0))  # 转换为 HWC 格式
            img = (img + 1) / 2  # 反归一化到 [0, 1]
            ax.imshow(img)
            ax.set_title(f'Pred: {class_names[predicted[i].item()]}\nTrue: {class_names[test_labels[i].item()]}')
            ax.axis('off')
        plt.suptitle(f'{dataset_name} Predictions')
        plt.tight_layout()
        plt.show()

# 类别名称
fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

visualize_predictions(fashion_model, fashion_test_loader, 'FashionMNIST', fashion_classes)
visualize_predictions(cifar_model, cifar_test_loader, 'CIFAR-10', cifar_classes)