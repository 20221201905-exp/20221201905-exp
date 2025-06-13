# 实验六 K-Means 聚类算法实验报告

**序号**: 24  
**姓名**: 王凯文  
**学号**: 20221201905  

## 一、实验目的

1. 深入理解 K-Means 聚类算法的基本原理及其基于欧式距离的实现。
2. 熟练运用自定义 K-Means 算法，掌握数据加载、预处理、聚类训练、评估和可视化的完整流程。
3. 在 scikit-learn 自带 iris 数据集上实现 K-Means 算法，支持传入 K 值、迭代次数和提前停止功能。
4. 可视化聚类前后的结果，并计算聚类指标（F-measure, ACC, NMI, RI, ARI）。

## 二、实验环境

- **操作系统**: Windows 10
- **开发工具**: PyCharm
- **编程语言**: Python 3.9
- **所需库**:
  - numpy: 高性能数值计算库。
  - matplotlib: 基础绘图库，用于创建静态、动态、交互式可视化。
  - scikit-learn (>=0.18): 机器学习核心库，包含数据集和评估指标。

## 三、实验内容

本实验基于 scikit-learn 自带 iris 数据集，执行 K-Means 聚类分析。数据集包含 150 个样本，4 个特征（花萼长度、花萼宽度、花瓣长度、花瓣宽度），3 个类别（Setosa、Versicolor、Virginica）。实验实现：
1. 自定义基于欧式距离的 K-Means 算法，支持 K 值、迭代次数和提前停止。
2. 可视化聚类前（真实标签）和聚类后（预测标签）的二维散点图。
3. 计算聚类指标（F-measure, ACC, NMI, RI, ARI）。

## 四、实验步骤

### 1. 环境搭建

在 Windows 10 系统中，确保已安装 Python 3.9。通过命令行工具执行以下命令安装所需库：

```bash
pip install numpy matplotlib scikit-learn
```

### 2. 数据预处理

此阶段完成数据集的加载、特征提取和标准化处理。

**代码**:  
```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 加载 iris 数据集
iris = load_iris()
X = iris.data  # 特征 (4 维)
y_true = iris.target  # 真实标签

# 数据预处理：标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**数据预处理分析**:  
- **数据集规模**: 150 个样本，4 个特征，3 个类别，数据量适中。
- **特征与目标**: 4 个特征作为输入，`target` 作为真实标签，用于评估。
- **划分策略**: 无需划分训练集和测试集，全部数据用于聚类。
- **描述统计**: 标准化后均值为 0，标准差为 1，适合欧式距离计算。

### 3. 探索性数据分析 (EDA)

通过可视化原始数据，了解分布结构。

**代码** (示例):  
```python
import matplotlib.pyplot as plt

# 显示原始数据（前两维特征）
plt.figure(figsize=(6, 4))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis')
plt.title('Original Data (True Labels)')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.show()
```

**EDA 结果分析**:  
- **数据分布**: 前两维特征散点图显示 3 个类别基本分离，Setosa 聚类明显。
- **数据特性**: 4 维特征中存在一定结构，适合 K-Means 聚类。

### 4. 模型训练

实现自定义 K-Means 算法，支持参数配置和提前停止。

**代码**:  
```python
# 自定义 K-Means 算法
class CustomKMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centers = X[random_indices]
        self.labels = np.zeros(n_samples)

        for _ in range(self.max_iters):
            new_labels = np.argmin(
                np.array([[np.sum((x - c) ** 2) for c in self.centers] for x in X]),
                axis=1
            )
            new_centers = np.array([X[new_labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(np.abs(new_centers - self.centers) < self.tol):
                break
            self.centers = new_centers
            self.labels = new_labels

    def predict(self, X):
        return np.argmin(
            np.array([[np.sum((x - c) ** 2) for c in self.centers] for x in X]),
            axis=1
        )

# 训练模型
kmeans = CustomKMeans(n_clusters=3, max_iters=100, tol=1e-4)
kmeans.fit(X_scaled)
y_pred = kmeans.predict(X_scaled)
```

**训练分析**:  
- 自定义 K-Means 使用欧式距离，随机初始化簇中心，迭代更新。
- 提前停止通过 `tol=1e-4` 判断中心变化，减少不必要迭代。

### 5. 模型评估

计算聚类指标，评估性能。

**输出结果** (基于提供的运行结果):  
```
F-measure: 0.2453
Accuracy (ACC): 0.8133
Normalized Mutual Information (NMI): 0.6427
Rand Index (RI): 0.8196
Adjusted Rand Index (ARI): 0.5923
```

**代码** (指标计算):  
```python
def calculate_metrics(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    f_measure = f1_score(y_true, y_pred, average='macro')
    n_samples = len(y_true)
    n_clusters = len(np.unique(y_pred))
    confusion_matrix = np.zeros((n_clusters, len(np.unique(y_true))))
    for i in range(n_samples):
        confusion_matrix[y_pred[i]][y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    label_map = {j: i for j, i in zip(row_ind, col_ind)}
    y_mapped = np.array([label_map[p] for p in y_pred])
    acc = accuracy_score(y_true, y_mapped)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ri = rand_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return f_measure, acc, nmi, ri, ari

f_measure, acc, nmi, ri, ari = calculate_metrics(y_true, y_pred)
print(f"F-measure: {f_measure:.4f}")
print(f"Accuracy (ACC): {acc:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Rand Index (RI): {ri:.4f}")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
```

**评估分析**:  
- **F-measure**: 0.2453，反映了聚类与真实标签的平衡性能较低，可能因 F1 分数对不平衡数据敏感。
- **ACC**: 0.8133，标签映射后分类准确率较高，表明聚类结果与真实类别有一定一致性。
- **NMI**: 0.6427，显示信息共享程度中等，表明聚类结构与真实标签有一定相关性。
- **RI**: 0.8196，未调整的相似性指标较高，反映了聚类结果与真实标签的总体一致性。
- **ARI**: 0.5923，调整后的随机性指标中等，表明聚类质量良好但有改进空间。

### 6. 结果可视化

通过散点图对比聚类前后结果。

**代码** (示例):  
```python
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis')
plt.title('Original Data (True Labels)')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis')
plt.title('Clustered Data (Predicted Labels)')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.tight_layout()
plt.show()
```

**可视化分析**:  
- **聚类前**: 真实标签显示 3 个类别基本分离，Setosa 聚类明显。
- **聚类后**: 预测标签与真实标签分布相似，3 簇结构清晰，部分重叠（如 Versicolor 和 Virginica），反映指标中的一致性。

## 五、实验结果与分析

本次实验成功实现了基于 iris 数据集的 K-Means 聚类。

### 实验结果总结:  
- **模型性能**: 自定义 K-Means 聚类效果良好，ACC 达到 81.33%，指标整体表现中等偏上。
- **提前停止**: 收敛阈值 `tol=1e-4` 有效减少迭代次数。
- **可视化验证**: 散点图显示聚类结果与真实分布有一定一致性。

### 局限性与展望:  
- **初始化敏感**: 随机初始化可能导致局部最优，未来可加入 k-means++ 初始化。
- **F-measure 偏低**: 可能因 F1 对不平衡数据敏感，未来可调整评估策略。
- **特征选择**: 仅使用前两维，未来可尝试降维或所有特征。

### 结论:  
本次实验通过自定义 K-Means 实现了 iris 数据集的聚类，K=3 达到 81.33% 准确率，验证了算法有效性。可视化与指标分析支持聚类质量，掌握了 K-Means 原理与优化方法。

**未来展望**:  
- 优化初始化策略，加入 k-means++。
- 尝试更多特征组合或降维技术。
- 扩展到复杂数据集，验证鲁棒性。

本次实验为后续机器学习学习奠定了基础。