import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# 加载MNIST数据集
digits = load_digits()
X = digits.data
y = digits.target

# 数据归一化处理
X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))

# 使用PCA降维到2维方便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

# K-means进行聚类
kmeans = KMeans(n_clusters=10, random_state=42)
y_pred = kmeans.fit_predict(X_norm)

# 可视化聚类结果
labels = kmeans.labels_
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title("K-means Clustering of MNIST Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.show()

# 输出分类准确度
accuracy = accuracy_score(y, y_pred)
print("分类准确度：", accuracy)
