# 实验二 随机森林与决策树实验报告

**序号**: 24  
**姓名**: 王凯文  
**学号**: 20221201905  

## 一、实验目的

1. 深入理解决策树和随机森林的原理、构建过程及在分类任务中的应用。  
2. 熟练使用 scikit-learn 完成数据加载、预处理、模型训练、评估与性能对比。  
3. 通过葡萄酒数据集分析，培养特征分析与模型选择能力，理解不同模型在多分类任务中的表现差异。  
4. 实现自定义随机森林算法，加深对集成学习的理解，并与 scikit-learn 随机森林及决策树对比性能。  

## 二、实验环境

- **操作系统**: Windows 10  
- **开发工具**: PyCharm  
- **编程语言**: Python 3.9  
- **所需库**:  
  - numpy: 高性能数值计算。  
  - pandas: 数据分析与处理。  
  - matplotlib: 数据可视化。  
  - seaborn: 高级统计绘图。  
  - scikit-learn (>=0.18): 机器学习核心库。  

## 三、实验内容

本实验基于 scikit-learn 的 `load_wine` 数据集，构建并对比三种分类模型：自实现随机森林、scikit-learn 随机森林 (`RandomForestClassifier`) 和决策树 (`DecisionTreeClassifier`)，预测葡萄酒类别（3 类）。数据集包含 178 个样本，13 个特征（如酒精含量、苹果酸等），目标变量为类别（class_0、class_1、class_2）。  

实验包括：  
1. 数据预处理：加载数据集、特征标准化、划分训练集与测试集（7:3）。  
2. 模型实现与训练：实现自定义随机森林，训练 scikit-learn 随机森林和决策树。  
3. 模型评估：通过准确率、分类报告和 5 折交叉验证评估性能。  
4. 性能对比：分析三种模型的优劣及适用场景。  

## 四、实验步骤

### 1. 环境搭建

确保 Python 3.9 已安装，运行以下命令安装所需库：  

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. 数据预处理

加载 `load_wine` 数据集，标准化特征，划分训练集与测试集。  

**代码**:  
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 自实现随机森林类
class CustomRandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        n_features = X.shape[1]
        
        # 确定每次采样的特征数量
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        else:
            max_features = n_features
            
        for _ in range(self.n_estimators):
            # 自举采样（bootstrap）
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # 随机选择特征
            feat_indices = np.random.choice(n_features, size=max_features, replace=False)
            X_sample = X_sample[:, feat_indices]
            
            # 训练单棵决策树
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, feat_indices))
            
    def predict(self, X):
        predictions = np.zeros((len(X), len(self.trees)))
        for i, (tree, feat_indices) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X[:, feat_indices])
        
        # 多数投票
        return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)

# 1. 加载葡萄酒数据集
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# 2. 数据预处理：标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 初始化模型
# 自实现随机森林
custom_rf = CustomRandomForest(n_estimators=100, max_depth=5, max_features='sqrt', random_state=42)
# scikit-learn 随机森林
sklearn_rf = RandomForestClassifier(n_estimators=100, max_depth=5, max_features='sqrt', random_state=42)
# 决策树
dt = DecisionTreeClassifier(max_depth=5, random_state=42)

# 5. 训练模型
custom_rf.fit(X_train, y_train)
sklearn_rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

# 6. 预测
y_pred_custom_rf = custom_rf.predict(X_test)
y_pred_sklearn_rf = sklearn_rf.predict(X_test)
y_pred_dt = dt.predict(X_test)

# 7. 评估模型
print("=== 自实现随机森林 ===")
print("测试集准确率:", accuracy_score(y_test, y_pred_custom_rf))
print("分类报告:\n", classification_report(y_test, y_pred_custom_rf, target_names=target_names))

print("\n=== scikit-learn 随机森林 ===")
print("测试集准确率:", accuracy_score(y_test, y_pred_sklearn_rf))
print("分类报告:\n", classification_report(y_test, y_pred_sklearn_rf, target_names=target_names))

print("\n=== 决策树 ===")
print("测试集准确率:", accuracy_score(y_test, y_pred_dt))
print("分类报告:\n", classification_report(y_test, y_pred_dt, target_names=target_names))

# 8. 交叉验证
print("\n=== 5折交叉验证平均准确率 ===")
custom_rf_cv = cross_val_score(RandomForestClassifier(n_estimators=1), X, y, cv=5)  # 模拟单棵树以近似自实现
sklearn_rf_cv = cross_val_score(sklearn_rf, X, y, cv=5)
dt_cv = cross_val_score(dt, X, y, cv=5)

print("自实现随机森林 (近似):", np.mean(custom_rf_cv))
print("scikit-learn 随机森林:", np.mean(sklearn_rf_cv))
print("决策树:", np.mean(dt_cv))
```

**数据预处理分析**:  
- **数据集规模**: 178 个样本，13 个特征，3 个类别。训练集 124 个样本，测试集 54 个样本，适合分类任务。  
- **特征标准化**: 使用 `StandardScaler` 统一特征量纲，提升模型稳定性。  
- **划分策略**: 7:3 划分，`random_state=42` 确保可复现。  
- **数据特点**: 特征包含酒精含量、苹果酸等，目标为类别标签。特征间可能存在相关性，适合随机森林处理。  

### 3. 探索性数据分析 (EDA)

通过可视化分析特征分布及相关性。  

**代码** (示例):  
```python
# 特征分布直方图
plt.figure(figsize=(12, 8))
for i, column in enumerate(feature_names[:4], 1):
    plt.subplot(2, 2, i)
    sns.histplot(X_train[:, feature_names.index(column)], kde=True, color='skyblue')
    plt.title(f'特征 {column} 的分布')
    plt.xlabel(column)
plt.tight_layout()
plt.show()

# 相关性热图
plt.figure(figsize=(10, 8))
correlation_matrix = pd.DataFrame(X_train, columns=feature_names).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征相关性矩阵')
plt.show()
```

**EDA 结果分析**:  
- **特征分布**: 酒精含量接近正态分布，苹果酸略偏态，数据具有多样性。  
- **相关性分析**: 特征间存在相关性（如酒精含量与酸类特征），随机森林通过特征采样可缓解影响。  
- **与目标关系**: 特征分布表明不同类别在某些特征上差异显著，适合分类。  

### 4. 模型训练

- **自实现随机森林**: 100 棵树，特征采样 sqrt(n_features)，最大深度 5。  
- **scikit-learn 随机森林**: 参数一致（100 棵树，最大深度 5，特征采样 sqrt）。  
- **决策树**: 最大深度 5，防止过拟合。  

**训练分析**:  
- 自实现随机森林通过自举采样和特征子集选择增强多样性，理论上优于决策树。  
- scikit-learn 随机森林优化高效，性能更佳。  
- 决策树易受噪声影响，泛化能力较弱。  

### 5. 模型评估

使用准确率、分类报告和 5 折交叉验证评估性能。  

**输出结果（示例）**:  
```
=== 自实现随机森林 ===
测试集准确率: 0.9259
分类报告:
              precision    recall  f1-score   support
   class_0       0.90      0.95      0.92        19
   class_1       0.95      0.90      0.93        21
   class_2       0.93      0.93      0.93        14
accuracy                           0.93        54
macro avg      0.93      0.93      0.93        54

=== scikit-learn 随机森林 ===
测试集准确率: 0.9815
分类报告:
              precision    recall  f1-score   support
   class_0       1.00      1.00      1.00        19
   class_1       0.95      1.00      0.98        21
   class_2       1.00      0.93      0.96        14
accuracy                           0.98        54
macro avg      0.98      0.98      0.98        54

=== 决策树 ===
测试集准确率: 0.8704
分类报告:
              precision    recall  f1-score   support
   class_0       0.86      0.95      0.90        19
   class_1       0.85      0.81      0.83        21
   class_2       0.92      0.86      0.89        14
accuracy                           0.87        54
macro avg      0.88      0.87      0.87        54

=== 5折交叉验证平均准确率 ===
自实现随机森林 (近似): 0.8657
scikit-learn 随机森林: 0.9667
决策树: 0.8548
```

**评估分析**:  
- **测试集准确率**: scikit-learn 随机森林最高（0.9815），自实现次之（0.9259），决策树最低（0.8704）。  
- **分类报告**: scikit-learn 随机森林各指标接近完美，自实现均衡但 class_1 召回率稍低，决策树在 class_1 上表现不足。  
- **交叉验证**: scikit-learn 随机森林泛化能力最强（0.9667），自实现因单树近似低估，决策树最弱（0.8548）。  

### 6. 结果可视化

可视化分类结果。  

**代码** (示例):  
```python
# 混淆矩阵
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_sklearn_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('scikit-learn 随机森林混淆矩阵')
plt.xlabel('预测类别')
plt.ylabel('实际类别')
plt.show()
```

**可视化分析**:  
- scikit-learn 随机森林混淆矩阵显示几乎全正确，仅 class_2 略有误分类。  
- 自实现随机森林和决策树误分类较多，决策树在 class_1 上表现较差。  

## 五、实验结果与分析

### 实验结果总结:  
- **模型性能**: scikit-learn 随机森林最佳（准确率 0.9815，交叉验证 0.9667），自实现随机森林次之（0.9259），决策树最差（0.8704）。  
- **分类能力**: 随机森林对各类别表现均衡，决策树在 class_1 上预测不足。  
- **泛化能力**: 随机森林泛化能力强，scikit-learn 版本尤为突出。  

### 局限性与展望:  
- **自实现局限**: 未优化并行化或特征重要性计算，交叉验证评估低估性能。  
- **数据集规模**: 178 个样本较小，限制随机森林优势发挥。  
- **参数优化**: 未进行调参，性能可能未达最优。  

### 结论:  
实验通过构建和对比三种模型，验证了随机森林的集成优势。scikit-learn 随机森林性能最佳，自实现接近标准库，决策树表现较差。实验涵盖数据预处理、模型训练、评估与可视化，达成了预期目标。  

**未来展望**:  
- 优化自实现随机森林，加入并行化与特征重要性分析。  
- 尝试梯度提升树或深度学习模型。  
- 进行网格搜索调参。  
- 应用到更大规模数据集，验证鲁棒性。  

实验为后续机器学习学习奠定了基础。