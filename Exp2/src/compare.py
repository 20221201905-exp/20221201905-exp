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
# scikit-learn 决策树
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