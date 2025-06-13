import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# 加载 scikit-learn 自带 digits 数据集
digits = load_digits()
X = digits.data  # 特征 (8x8=64 像素值)
y = digits.target  # 标签 (0-9)

# 数据预处理：标准化和划分数据集
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 自定义距离度量函数
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def cosine_distance(x1, x2):
    return 1 - np.dot(x1, x2) / (np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2)))

# 自定义 KNN 算法
class CustomKNN:
    def __init__(self, k=3, distance_metric=euclidean_distance):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # 计算距离
            distances = [self.distance_metric(x, x_train) for x_train in self.X_train]
            # 获取 k 个最近邻的索引
            k_indices = np.argsort(distances)[:self.k]
            # 多数投票
            k_nearest_labels = self.y_train[k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        return np.array(predictions)

# 寻找最优 K 值
k_values = range(1, 31, 2)
accuracies = []
for k in k_values:
    knn = CustomKNN(k=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    if acc >= 0.90:
        print(f"K={k}, 准确率: {acc:.4f}")
optimal_k = k_values[np.argmax(accuracies)]
print(f"最优 K 值: {optimal_k}, 最大准确率: {max(accuracies):.4f}")

# 使用最优 K 值与不同距离度量
knn_euclidean = CustomKNN(k=optimal_k, distance_metric=euclidean_distance)
knn_manhattan = CustomKNN(k=optimal_k, distance_metric=manhattan_distance)
knn_cosine = CustomKNN(k=optimal_k, distance_metric=cosine_distance)

knn_euclidean.fit(X_train, y_train)
knn_manhattan.fit(X_train, y_train)
knn_cosine.fit(X_train, y_train)

y_pred_euclidean = knn_euclidean.predict(X_test)
y_pred_manhattan = knn_manhattan.predict(X_test)
y_pred_cosine = knn_cosine.predict(X_test)

print("\n自定义 KNN 结果:")
print(f"欧式距离准确率: {accuracy_score(y_test, y_pred_euclidean):.4f}")
print(f"曼哈顿距离准确率: {accuracy_score(y_test, y_pred_manhattan):.4f}")
print(f"余弦距离准确率: {accuracy_score(y_test, y_pred_cosine):.4f}")

# scikit-learn KNN 对比
sklearn_knn = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')
sklearn_knn.fit(X_train, y_train)
y_pred_sklearn = sklearn_knn.predict(X_test)
print(f"\nscikit-learn KNN 结果:")
print(f"K={optimal_k} 准确率: {accuracy_score(y_test, y_pred_sklearn):.4f}")

# 可视化 K 值与准确率的关系
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('K 值与准确率的关系')
plt.xlabel('K 值')
plt.ylabel('准确率')
plt.grid(True)
plt.show()