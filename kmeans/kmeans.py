import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target

# 由于MNIST数据集很大，为了加快运行速度，我们仅使用部分数据
X = X[:10000]
y = y[:10000]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K-means进行聚类
kmeans = KMeans(n_clusters=10, random_state=42)
y_pred = kmeans.fit_predict(X)

# 可视化聚类结果
labels = kmeans.labels_
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title("K-means Clustering of MNIST Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.show()


# 使用KMeans算法进行聚类
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_pre  dict(X)

# 获取聚类中心
centroids = kmeans.cluster_centers_

# 为了可视化，我们将每个聚类中心转换回28x28像素的图像
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
centers = centroids.reshape(10, 28, 28)
for ax, center in zip(axs.flat, centers):
    ax.set_axis_off()
    ax.imshow(center, interpolation='nearest', cmap='gray')

plt.show()
