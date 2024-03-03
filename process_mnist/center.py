import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F
import numpy as np

# 假设我们有一个已经训练好的Encoder模型
# 我们使用一部分训练数据来初始化聚类中心
def init_cluster_centers(encoder, trainloader, n_clusters=10):
    features = []
    for data, _ in trainloader:
        # 假设我们只使用部分数据来初始化聚类中心
        features.append(encoder(data).view(data.size(0), -1).detach().numpy())
        if len(features) > 100:  # 使用前100批数据来初始化聚类中心
            break
    features = np.concatenate(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    cluster_centers = kmeans.cluster_centers_
    return torch.tensor(cluster_centers, dtype=torch.float32)

# 加入聚类损失的训练过程
def train_with_clustering(model, trainloader, optimizer, cluster_centers, lambda_cluster=1.0, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in trainloader:
            optimizer.zero_grad()
            # 假设 inputs 的值在 [0, 255] 范围内
            inputs = inputs / 255.0  # 确保 inputs 在 [0, 1] 范围内
            outputs = model(inputs)
            # 计算重构损失
            reconstruction_loss = F.binary_cross_entropy(outputs, inputs, reduction='sum') / inputs.size(0)
            # 计算聚类损失
            features = model.encoder(inputs).view(inputs.size(0), -1)  # 获取特征表示
            dist = torch.cdist(features, cluster_centers)  # 计算到聚类中心的距离
            clustering_loss = torch.min(dist, dim=1)[0].mean()  # 选择最近的聚类中心，计算平均距离作为聚类损失
            # 总损失
            total_loss = reconstruction_loss + lambda_cluster * clustering_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')


# # 初始化聚类中心
# cluster_centers = init_cluster_centers(model.encoder, trainloader, n_clusters=10)
#
# # 使用聚类损失训练模型
# train_with_clustering(model, trainloader, optimizer, cluster_centers)
