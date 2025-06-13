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

# 1. 显示原始图像
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = X[i].reshape(28, 28)  # 转换为 28x28 图像
    plt.imshow(img, cmap='gray')
    plt.title(f'Label: {y[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# 2. PCA 转换与可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.title('PCA of MNIST Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

# 3. 多种降维方法可视化
methods = {
    'PCA': PCA(n_components=2),
    'Isomap': Isomap(n_components=2, n_neighbors=5),
    'LLE': LocallyLinearEmbedding(n_components=2, n_neighbors=5),
    'LE': SpectralEmbedding(n_components=2, n_neighbors=5),
    't-SNE': TSNE(n_components=2, random_state=42)
}

plt.figure(figsize=(15, 10))
for i, (method_name, method) in enumerate(methods.items(), 1):
    X_transformed = method.fit_transform(X_scaled[:1000])  # 取前 1000 样本以减少计算量
    plt.subplot(2, 3, i)
    scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y[:1000], cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Digit')
    plt.title(f'{method_name} of MNIST')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
plt.tight_layout()
plt.show()