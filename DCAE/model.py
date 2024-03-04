import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DCAE(nn.Module):
    def __init__(self):
        super(DCAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=10, stride=1, padding=0),  # input is 1 x 10, output is 32 x 1
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=10, stride=1, padding=0),  # output is 64 x 1
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=10, stride=1, padding=0),  # output is 128 x 1
            nn.ReLU(True),
            nn.Conv1d(128, 128, kernel_size=10, stride=1, padding=0),  # output is 128 x 1
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128, 384),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 1)),
            nn.ConvTranspose1d(128, 128, kernel_size=10, stride=1, padding=0),  # output is 128 x 1
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=10, stride=1, padding=0),  # output is 64 x 1
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=10, stride=1, padding=0),  # output is 32 x 1
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 1, kernel_size=10, stride=1, padding=0),  # output is 1 x 10
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Instantiate the model
model = DCAE()
# Print the model
print(model)





# 假设 h(x) 是自编码器的编码器部分的输出，c_k 是聚类中心，这些将在训练过程中更新
# h(x) 的尺寸假设为 (batch_size, num_features)，c_k 的尺寸假设为 (num_clusters, num_features)


def deep_clustering_loss(reconstructed_x, x, encoded_features, cluster_centers, lambda_val=0.1):
    """
    :param reconstructed_x: The output of the autoencoder's decoder. This is the reconstruction of the input data after it has been encoded to a lower-dimensional space and then decoded back to the original space.
    :param x: The original input data to the autoencoder.
    :param encoded_features: The output of the autoencoder's encoder. These are the features in the lower-dimensional space, which are used for clustering.
    :param cluster_centers: The current centers of the clusters, which should be updated periodically using an algorithm like k-means.
    :param lambda_val: A hyperparameter that balances the reconstruction loss and the clustering loss. It determines the weight of the clustering loss in the total loss function.
    For instance, you might start with values like 0.1, 1, and 10, and then adjust up or down depending on whether the model performance indicates that the clustering loss or the reconstruction loss needs to be weighted more heavily.
    reconstructed_x：自编码器解码器的输出。这是输入数据经过编码到低维空间然后再解码回原始空间后的重构数据。
    x：自编码器的原始输入数据。
    encoded_features：自编码器编码器的输出。这些是低维空间中的特征，用于聚类。
    cluster_centers：聚类中心的当前位置，应该定期使用像k-means这样的算法进行更新。
    lambda_val：一个超参数，用于平衡重构损失和聚类损失。它决定了聚类损失在总损失函数中的权重。
    :return:
    """
    # 自编码器重构损失
    reconstruction_loss = F.mse_loss(reconstructed_x, x)

    # 计算每个编码特征和它的最近聚类中心之间的距离
    clustering_loss = torch.mean((encoded_features.unsqueeze(1) - cluster_centers) ** 2, dim=2).min(dim=1)[0].mean()

    # 总损失
    total_loss = reconstruction_loss + lambda_val * clustering_loss

    return total_loss, reconstruction_loss, clustering_loss


def init_cluster_center(encoded_features_np):
    from sklearn.cluster import KMeans
    import torch

    # 假设 encoded_features 是你的模型编码器的输出特征，形状为 (n_samples, n_features)
    # 将 PyTorch 张量转换为 NumPy 数组，以适用于 sklearn 的 k-means
    encoded_features_np = encoded_features.detach().cpu().numpy()

    # 选择聚类的数量
    n_clusters = 10  # 例如，根据你的任务需求设置

    # 使用 k-means++ 初始化聚类中心
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    kmeans.fit(encoded_features_np)

    # 获取聚类中心并转换回 PyTorch 张量
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(encoded_features.device)


def update_cluster_center(encoded_features):
    # Assuming encoded_features is a 2D tensor of shape (num_samples, num_features)
    # and we want to cluster these features into k clusters

    # Convert PyTorch tensor to numpy array if necessary
    encoded_features_np = encoded_features.cpu().detach().numpy()

    # Define the number of clusters
    num_clusters = 10  # for example, set to the number of classes in CIFAR-100

    # Initialize the KMeans algorithm with the desired number of clusters
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)

    # Fit the KMeans algorithm on the encoded features to compute cluster centers
    kmeans.fit(encoded_features_np)

    # Retrieve the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Convert cluster centers back to PyTorch tensor and send to the device
    cluster_centers_tensor = torch.tensor(cluster_centers).to(encoded_features.device)
    # Now you can use cluster_centers_tensor in your loss function
    return cluster_centers_tensor


optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 10
processed_dataset = np.load('processed_dataset.npy')
dataloader = torch.tensor(processed_dataset)
# 训练循环
for epoch in range(num_epochs):
    for data in dataloader:
        optimizer.zero_grad()
        # 输入数据
        x = data
        # 前向传播
        reconstructed_x = model(x)
        encoded_features = model.encoder(x)
        cluster_centers = init_cluster_center(encoded_features)
        # 计算损失
        lambda_val = 0.1
        loss, rec_loss, clust_loss = deep_clustering_loss(reconstructed_x, x, encoded_features,
                                                          cluster_centers, lambda_val)
        # 后向传播
        loss.backward()
        # 更新权重
        optimizer.step()
    cluster_centers = update_cluster_center(cluster_centers)

    # Now you can use cluster_centers in your loss function
    # 更新聚类中心的逻辑应该在这里添加，通常是使用k-means或者其它聚类算法
# 注意：聚类中心的更新通常不是每个训练步骤都进行，可能是每个epoch后或者在特定的迭代次数后进行。
