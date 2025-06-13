import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, rand_score, adjusted_rand_score

# 自定义 K-Means 算法
class CustomKMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol  # 提前停止的收敛阈值

    def fit(self, X):
        n_samples, n_features = X.shape
        # 随机初始化簇中心
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centers = X[random_indices]
        self.labels = np.zeros(n_samples)

        for _ in range(self.max_iters):
            # 分配样本到最近的簇
            new_labels = np.argmin(
                np.array([[np.sum((x - c) ** 2) for c in self.centers] for x in X]),
                axis=1
            )
            # 更新簇中心
            new_centers = np.array([X[new_labels == k].mean(axis=0) for k in range(self.n_clusters)])
            # 检查收敛
            if np.all(np.abs(new_centers - self.centers) < self.tol):
                break
            self.centers = new_centers
            self.labels = new_labels

    def predict(self, X):
        return np.argmin(
            np.array([[np.sum((x - c) ** 2) for c in self.centers] for x in X]),
            axis=1
        )

# 加载数据集
iris = load_iris()
X = iris.data
y_true = iris.target  # 真实标签，用于评估

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练自定义 K-Means
kmeans = CustomKMeans(n_clusters=3, max_iters=100, tol=1e-4)
kmeans.fit(X_scaled)

# 预测标签
y_pred = kmeans.predict(X_scaled)

# 计算聚类指标
def calculate_metrics(y_true, y_pred):
    # F-measure (使用 macro average)
    f_measure = f1_score(y_true, y_pred, average='macro')
    # ACC (需要标签映射)
    def map_labels(y_true, y_pred):
        from scipy.optimize import linear_sum_assignment
        n_samples = len(y_true)
        n_clusters = len(np.unique(y_pred))
        confusion_matrix = np.zeros((n_clusters, len(np.unique(y_true))))
        for i in range(n_samples):
            confusion_matrix[y_pred[i]][y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        label_map = {j: i for j, i in zip(row_ind, col_ind)}
        return np.array([label_map[p] for p in y_pred])
    y_mapped = map_labels(y_true, y_pred)
    acc = accuracy_score(y_true, y_mapped)
    # NMI
    nmi = normalized_mutual_info_score(y_true, y_pred)
    # RI
    ri = rand_score(y_true, y_pred)
    # ARI
    ari = adjusted_rand_score(y_true, y_pred)
    return f_measure, acc, nmi, ri, ari

f_measure, acc, nmi, ri, ari = calculate_metrics(y_true, y_pred)
print(f"F-measure: {f_measure:.4f}")
print(f"Accuracy (ACC): {acc:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Rand Index (RI): {ri:.4f}")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")

# 可视化
plt.figure(figsize=(12, 5))

# 聚类前 (原始数据)
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis')
plt.title('Original Data (True Labels)')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')

# 聚类后
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis')
plt.title('Clustered Data (Predicted Labels)')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')

plt.tight_layout()
plt.show()