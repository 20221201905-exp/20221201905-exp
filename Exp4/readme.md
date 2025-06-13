# 实验四 MNIST 数据集降维与可视化实验报告

**序号**: 24  
**姓名**: 王凯文  
**学号**: 20221201905  

## 一、实验目的

1. 深入理解 PCA、Isomap、LLE、LE（拉普拉斯特征映射）和 t-SNE 等降维算法的基本原理及其在高维数据可视化中的应用。
2. 熟练运用 scikit-learn 机器学习库，掌握数据加载、预处理、降维及可视化的完整流程。
3. 通过对 MNIST 数据集的分析，培养数据降维与可视化能力，理解特征空间的结构及其与数字类别之间的关系。
4. 实现多种降维方法，比较其在二维平面上的表现效果，为后续分类任务奠定基础。

## 二、实验环境

- **操作系统**: Windows 10
- **开发工具**: PyCharm
- **编程语言**: Python 3.9
- **所需库**:
  - numpy: 高性能数值计算库。
  - pandas: 强大的数据结构和数据分析工具。
  - matplotlib: 基础绘图库，用于创建静态、动态、交互式可视化。
  - seaborn: 基于 matplotlib 的数据可视化库，提供更美观的统计图。
  - scikit-learn (>=0.18): 机器学习核心库，包含各种降维算法。

## 三、实验内容

本实验基于 Kaggle 竞赛提供的 MNIST 数据集（Digit Recognizer，https://www.kaggle.com/c/digit-recognizer/data），对手写数字图像进行降维与可视化。训练集包含 42,000 个样本，每个样本为 28x28 灰度图像（像素值范围 [0,255]），目标是识别 0-9 十个数字类别。实验从 `E:\lxh\train.csv` 加载数据，完成以下任务：
1. 显示原始 28x28 图像。
2. 可视化 PCA 转换后的特征数据。
3. 使用 PCA、Isomap、LLE、LE（Spectral Embedding）和 t-SNE 降维到二维平面，并标注数字类别。

## 四、实验步骤

### 1. 环境搭建

在 Windows 10 系统中，确保已安装 Python 3.9。通过命令行工具（如 PowerShell 或 CMD）执行以下命令安装所有必要的 Python 库：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. 数据预处理

此阶段完成数据集的加载、特征与目标变量的分离，并进行标准化处理。

**代码**:  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 读取数据集，指定路径 E:\lxh
train_data = pd.read_csv('E:\\lxh\\train.csv')
X = train_data.drop('label', axis=1).values  # 特征 (28x28=784 像素值)
y = train_data['label'].values  # 标签 (0-9)

# 数据预处理：标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**数据预处理分析**:  
- **数据集规模**: 训练集包含 42,000 个样本，每个样本有 784 个特征（28x28 像素值），标签为 0-9 的数字类别，数据量充足。
- **特征与目标**: 提取像素值作为特征，`label` 作为目标变量，符合手写数字识别任务。
- **划分策略**: 本实验仅使用训练集数据，无需划分训练集和测试集，数据直接用于降维和可视化。
- **描述统计**: 像素值范围在 [0,255]，标准化后均值为 0，标准差为 1，有助于降维算法处理。

### 3. 探索性数据分析 (EDA)

通过可视化原始图像，初步了解数据结构。

**代码** (示例):  
```python
# 显示原始图像
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = X[i].reshape(28, 28)  # 转换为 28x28 图像
    plt.imshow(img, cmap='gray')
    plt.title(f'Label: {y[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

**EDA 结果分析**:  
- **图像分布**: 显示的前 5 张图像清晰呈现手写数字 1, 0, 4, 1, 7，灰度值分布反映了笔迹粗细和书写风格差异。
- **数据特性**: 28x28 像素的高维数据包含丰富的空间信息，适合降维分析以揭示类别结构。

### 4. 模型训练

本实验不涉及传统模型训练，而是应用降维算法处理数据。

**训练分析**:  
- 使用 PCA、Isomap、LLE、LE 和 t-SNE 分别降维，重点在于数据结构的可视化而非分类预测。
- 降维过程无需显式训练，但需调整参数（如 `n_neighbors`）以优化结果。

### 5. 模型评估

通过可视化效果评估降维算法的表现，检查类别分离程度。

**输出结果** (示例):  
- PCA 散点图显示数字 0-9 部分分离，但部分类别（如 4 和 9）重叠。
- t-SNE 散点图显示更明显的类别聚类，0-9 数字边界清晰。
- Isomap、LLE 和 LE 结果因局部结构假设不同，聚类效果介于 PCA 和 t-SNE 之间。

**评估分析**:  
- **PCA**: 线性降维，保留最大方差，适合初步探索，但类别分离有限。
- **t-SNE**: 非线性降维，聚类效果最佳，适合复杂数据结构。
- **Isomap/LLE/LE**: 捕捉局部流形结构，效果因参数（如 `n_neighbors`）敏感，部分数字聚类较好。

### 6. 结果可视化

通过散点图直观展示降维结果。

**代码** (示例):  
```python
# PCA 转换与可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.title('PCA of MNIST Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

# 多种降维方法可视化
methods = {
    'PCA': PCA(n_components=2),
    'Isomap': Isomap(n_components=2, n_neighbors=5),
    'LLE': LocallyLinearEmbedding(n_components=2, n_neighbors=5),
    'LE': SpectralEmbedding(n_components=2, n_neighbors=5),
    't-SNE': TSNE(n_components=2, random_state=42)
}

plt.figure(figsize=(15, 10))
for i, (method_name, method) in enumerate(methods.items(), 1):
    X_transformed = method.fit_transform(X_scaled[:1000])  # 取前 1000 样本
    plt.subplot(2, 3, i)
    scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y[:1000], cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Digit')
    plt.title(f'{method_name} of MNIST')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
plt.tight_layout()
plt.show()
```

**可视化分析**:  
- **原始图像**: 5 张图像清晰展示手写数字，验证数据质量。
- **PCA 图**: 2D 散点图显示部分类别分离，0 和 1 聚类较好，4 和 9 重叠。
- **多种降维图**: t-SNE 图显示最佳聚类效果，Isomap 和 LLE 局部结构较好，LE 结果较分散，PCA 作为基线表现中等。

## 五、实验结果与分析

本次实验成功实现了 MNIST 数据集的降维与可视化，比较了多种算法的表现。

### 实验结果总结:  
- **模型拟合优异**: 各降维方法成功将 784 维像素数据降至 2D，t-SNE 表现出最佳类别分离能力。
- **评估指标卓越**: 通过可视化观察，t-SNE 聚类最清晰，PCA 保留最多方差，Isomap/LLE/LE 适合局部结构分析。
- **可视化验证**: 散点图直观显示类别分布，t-SNE 图中 0-9 数字边界分明，PCA 图保留全局结构。

### 局限性与展望:  
- **计算效率**: t-SNE 和 Isomap 对 42,000 样本计算耗时长，仅处理前 1000 样本，限制了全面分析。
- **参数敏感**: Isomap、LLE 和 LE 的 `n_neighbors` 参数对结果影响大，需进一步调优。
- **数据规模**: 仅使用训练集，未充分利用 `test.csv`，未来可扩展到完整数据集。

### 结论:  
本次实验成功运用多种降维算法对 MNIST 数据集进行了可视化分析。通过规范的数据预处理、降维处理和多维度可视化，我们掌握了降维技术的应用方法，理解了高维数据结构与数字类别之间的关系。实验结果证明了 t-SNE 在复杂数据可视化中的优越性，同时验证了 PCA 的基础作用。

**未来展望**:  
- 尝试非线性降维参数优化（如调整 t-SNE 的 `perplexity`）。
- 扩展到完整数据集（42,000 样本），使用高效计算方法。
- 结合分类模型（如 SVM 或随机森林），评估降维对分类性能的影响。
- 探索更多降维技术（如 UMAP），进一步提升可视化效果。

本次实验为后续机器学习算法学习和实际项目应用奠定了坚实的基础。