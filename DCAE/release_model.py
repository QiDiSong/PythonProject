import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

feature_length = 784
in_channels = 1
# 根据截屏中描述的网络结构定义DCAE模型
class DCAE(nn.Module):
    def __init__(self):
        super(DCAE, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (feature_length - 9 * 4), 384),
            nn.ReLU(),
            nn.Linear(384, 10),
            nn.ReLU()
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(10, 384),
            nn.ReLU(),
            nn.Linear(384, 128 * (feature_length - 9 * 4)),
            nn.ReLU(),
            nn.Unflatten(1, (128, feature_length - 9 * 4)),
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=in_channels, kernel_size=10, stride=1),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # x = torch.tanh(x)  # Ensure the output is in [-1, 1]
        return x


# 创建模型实例
model = DCAE()

# 检查模型是否按预期构建
print(model)

# 创建一个示例输入张量
example_input = torch.randn(17, in_channels, 784)

# 将输入传递给模型并获得输出
output = model(example_input)
print("Output shape:", output.shape)

# MNIST数据集的预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1))
])
# 加载MNIST数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# 定义损失函数和优化器
criterion_mse = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 检查CUDA支持并定义设备变量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 确保模型使用定义的设备
model.to(device)

if __name__ == '__main__':
    # 训练模型
    num_epochs = 10  # 为了快速演示，实际可能需要更多的epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.view(-1, 1, 784)  # 调整输入以匹配模型的期望形状

            optimizer.zero_grad()

            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            loss = criterion_mse(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print("Epoch:", epoch + 1, "Batch: ", batch, " Loss:", running_loss / (batch + 1))
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

    print('Finished Training')
    PATH = './dcae_encoder.pth'
    torch.save(model.state_dict(), PATH)





# # 选择部分数据进行测试和可视化
# dataiter = iter(testloader)
# images, labels = next(dataiter)
# images = images.view(-1, 1, 784)  # 调整形状以匹配模型输入
# outputs = model(images)
#
# # 可视化第一张图像和其重建结果
# plt.figure(figsize=(9, 3))
# plt.subplot(1, 2, 1)
# plt.imshow(images[0].view(28, 28).detach().numpy(), cmap='gray')
# plt.title('Original Image')
# plt.subplot(1, 2, 2)
# plt.imshow(outputs[0].view(28, 28).detach().numpy(), cmap='gray')
# plt.title('Reconstructed Image')
# plt.show()

if __name__ == '__main__':
    # 训练模型
    num_epochs = 10  # 为了快速演示，实际可能需要更多的epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.view(-1, 1, 784)  # 调整输入以匹配模型的期望形状

            optimizer.zero_grad()

            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            loss = criterion_mse(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print("Epoch:", epoch + 1, "Batch: ", batch, " Loss:", running_loss / (batch + 1))
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

    print('Finished Training')
    PATH = './dcae_encoder.pth'
    torch.save(model.state_dict(), PATH)





# # 选择部分数据进行测试和可视化
# dataiter = iter(testloader)
# images, labels = next(dataiter)
# images = images.view(-1, 1, 784)  # 调整形状以匹配模型输入
# outputs = model(images)
#
# # 可视化第一张图像和其重建结果
# plt.figure(figsize=(9, 3))
# plt.subplot(1, 2, 1)
# plt.imshow(images[0].view(28, 28).detach().numpy(), cmap='gray')
# plt.title('Original Image')
# plt.subplot(1, 2, 2)
# plt.imshow(outputs[0].view(28, 28).detach().numpy(), cmap='gray')
# plt.title('Reconstructed Image')
# plt.show()





#
# # # Assuming 'trainloader' is a DataLoader for the MNIST dataset
# # for images, labels in trainloader:
# #     # Flatten the batch of images
# #     images = images.unsqueeze(1)
# #     print(images.shape)  # This should print torch.Size([64, 784])
# #     # break  # Breaking after the first batch to demonstrate
#
#
# def deep_clustering_loss(reconstructed_x, x, encoded_features, cluster_centers, lambda_val=0.1):
#     """
#     :param reconstructed_x: The output of the autoencoder's decoder. This is the reconstruction of the input data after it has been encoded to a lower-dimensional space and then decoded back to the original space.
#     :param x: The original input data to the autoencoder.
#     :param encoded_features: The output of the autoencoder's encoder. These are the features in the lower-dimensional space, which are used for clustering.
#     :param cluster_centers: The current centers of the clusters, which should be updated periodically using an algorithm like k-means.
#     :param lambda_val: A hyperparameter that balances the reconstruction loss and the clustering loss. It determines the weight of the clustering loss in the total loss function.
#     For instance, you might start with values like 0.1, 1, and 10, and then adjust up or down depending on whether the model performance indicates that the clustering loss or the reconstruction loss needs to be weighted more heavily.
#     reconstructed_x：自编码器解码器的输出。这是输入数据经过编码到低维空间然后再解码回原始空间后的重构数据。
#     x：自编码器的原始输入数据。
#     encoded_features：自编码器编码器的输出。这些是低维空间中的特征，用于聚类。
#     cluster_centers：聚类中心的当前位置，应该定期使用像k-means这样的算法进行更新。
#     lambda_val：一个超参数，用于平衡重构损失和聚类损失。它决定了聚类损失在总损失函数中的权重。
#     :return:
#     """
#     # 自编码器重构损失
#     reconstruction_loss = F.mse_loss(reconstructed_x, x)
#     # 计算每个编码特征和它的最近聚类中心之间的距离
#     clustering_loss = torch.mean((encoded_features.unsqueeze(1) - cluster_centers) ** 2, dim=2).min(dim=1)[0].mean()
#     # 总损失
#     total_loss = reconstruction_loss + lambda_val * clustering_loss
#     return total_loss, reconstruction_loss, clustering_loss
#
#
# def init_cluster_center(encoded_features_np):
#     from sklearn.cluster import KMeans
#     import torch
#
#     # 假设 encoded_features 是你的模型编码器的输出特征，形状为 (n_samples, n_features)
#     # 将 PyTorch 张量转换为 NumPy 数组，以适用于 sklearn 的 k-means
#     encoded_features_np = encoded_features.detach().cpu().numpy()
#
#     # 选择聚类的数量
#     n_clusters = 10  # 例如，根据你的任务需求设置
#
#     # 使用 k-means++ 初始化聚类中心
#     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
#     kmeans.fit(encoded_features_np)
#
#     # 获取聚类中心并转换回 PyTorch 张量
#     return torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(encoded_features.device)
#
#
# def update_cluster_center(encoded_features):
#     # Assuming encoded_features is a 2D tensor of shape (num_samples, num_features)
#     # and we want to cluster these features into k clusters
#
#     # Convert PyTorch tensor to numpy array if necessary
#     encoded_features_np = encoded_features.cpu().detach().numpy()
#
#     # Define the number of clusters
#     num_clusters = 10  # for example, set to the number of classes in CIFAR-100
#
#     # Initialize the KMeans algorithm with the desired number of clusters
#     kmeans = KMeans(n_clusters=num_clusters, n_init=10)
#
#     # Fit the KMeans algorithm on the encoded features to compute cluster centers
#     kmeans.fit(encoded_features_np)
#
#     # Retrieve the cluster centers
#     cluster_centers = kmeans.cluster_centers_
#
#     # Convert cluster centers back to PyTorch tensor and send to the device
#     cluster_centers_tensor = torch.tensor(cluster_centers).to(encoded_features.device)
#     # Now you can use cluster_centers_tensor in your loss function
#     return cluster_centers_tensor
#
#
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# num_epochs = 10
# # processed_dataset = np.load('processed_dataset.npy')
# # dataloader = torch.tensor(processed_dataset)
# # 训练循环
# for epoch in range(num_epochs):
#     for data, _ in trainloader:
#         optimizer.zero_grad()
#         # 输入数据
#         # Flatten the batch of images
#         x = data.unsqueeze(1)
#         # 前向传播
#         reconstructed_x = model(x)
#         encoded_features = model.encoder(x)
#         cluster_centers = init_cluster_center(encoded_features)
#         # 计算损失
#         lambda_val = 0.1
#         loss, rec_loss, clust_loss = deep_clustering_loss(reconstructed_x, x, encoded_features,
#                                                           cluster_centers, lambda_val)
#         print(epoch, loss.item())
#         # 后向传播
#         loss.backward()
#         # 更新权重
#         optimizer.step()
#     cluster_centers = update_cluster_center(cluster_centers)