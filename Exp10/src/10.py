import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

class Config:
    IMAGE_SIZE = (64, 64)
    BATCH_SIZE = 32  # 增大batch size以提高效率
    LEARNING_RATE = 0.001
    EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_PATH = r"D:/桌面/理塘/PetImages"
    NUM_CLASSES = 2
    EARLY_STOPPING_PATIENCE = 5  # 减少耐心值以更快停止

# 数据集类
class PetDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('L')  # 转换为灰度图（2D）
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"跳过无法读取的图像: {self.image_paths[idx]}")
            return self.__getitem__((idx + 1) % len(self.image_paths))

# 数据加载函数
def load_data(data_path):
    image_paths = []
    labels = []
    
    cat_path = os.path.join(data_path, 'Cat')
    if os.path.exists(cat_path):
        for filename in os.listdir(cat_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cat_path, filename))
                labels.append(0)
    
    dog_path = os.path.join(data_path, 'Dog')
    if os.path.exists(dog_path):
        for filename in os.listdir(dog_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(dog_path, filename))
                labels.append(1)
    
    print(f"总共加载了 {len(image_paths)} 张图像")
    print(f"猫的图像数量: {labels.count(0)}")
    print(f"狗的图像数量: {labels.count(1)}")
    
    return image_paths, labels

# ResNet基本块
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

# 简化ResNet模型（ResNet10）
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 32  # 减少初始通道数
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 输入通道改为1（灰度）
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet10(num_classes=2):
    return ResNet(BasicBlock, [1, 1, 1], num_classes)  # 更少的层

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*train_correct/train_total:.2f}%'})
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*val_correct/val_total:.2f}%'})
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print('-' * 60)
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, train_accuracies, val_losses, val_accuracies

# 测试函数
def test_model(model, test_loader):
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_predictions, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing')
        for images, labels in test_bar:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            test_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*test_correct/test_total:.2f}%'})
    
    test_accuracy = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    return test_accuracy, all_predictions, all_labels

# 可视化函数
def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 可视化预测结果
def visualize_predictions(model, test_loader, num_samples=16):
    model.eval()
    class_names = ['Cat', 'Dog']
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i // 4, i % 4]
        image = images[i].cpu().squeeze(0)  # 移除通道维度
        image = torch.clamp(image, 0, 1)
        ax.imshow(image, cmap='gray')
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        color = 'green' if labels[i] == predicted[i] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# 可视化标签分布
def visualize_label_distribution(labels):
    class_names = ['Cat', 'Dog']
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar([class_names[i] for i in unique], counts, color=['orange', 'skyblue'])
    plt.title('Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                str(count), ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ GPU不可用，使用CPU")
    print(f"使用设备: {Config.DEVICE}")
    print("-" * 50)
    
    # 数据变换（针对灰度图像）
    transform_train = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 灰度图像标准化
    ])

    transform_valid = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    print("正在加载数据...")
    image_paths, labels = load_data(Config.DATA_PATH)
    visualize_label_distribution(labels)
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"训练集大小: {len(train_paths)}")
    print(f"验证集大小: {len(val_paths)}")
    print(f"测试集大小: {len(test_paths)}")
    
    num_workers = 4 if Config.DEVICE.type == 'cuda' else 2
    train_dataset = PetDataset(train_paths, train_labels, transform_train)
    val_dataset = PetDataset(val_paths, val_labels, transform_valid)
    test_dataset = PetDataset(test_paths, test_labels, transform_valid)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    print("正在创建ResNet10模型...")
    model = resnet10(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    
    if Config.DEVICE.type == 'cuda':
        print("✅ 使用GPU训练，启用优化设置")
        torch.backends.cudnn.benchmark = True
    else:
        print("⚠️ 使用CPU训练，可能速度较慢")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print("开始训练模型...")
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, Config.EPOCHS
    )
    
    plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)
    
    print("正在测试模型...")
    test_accuracy, predictions, true_labels = test_model(model, test_loader)
    
    visualize_predictions(model, test_loader)
    
    torch.save(model.state_dict(), 'resnet10_cat_dog_classifier.pth')
    print("模型已保存为 'resnet10_cat_dog_classifier.pth'")
    
    print(f"\n最终测试准确率: {test_accuracy:.2f}%")
    if test_accuracy >= 90.0:
        print("✅ 成功达到90%以上的准确率!")
    else:
        print("❌ 未达到90%的准确率，可以尝试调整超参数或增加训练轮数")

if __name__ == "__main__":
    main()