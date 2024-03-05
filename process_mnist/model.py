import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from center import init_cluster_centers, train_with_clustering
import matplotlib.pyplot as plt
import numpy as np


# MNIST数据集的预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 加载MNIST数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Adjusted to accept 1 input channel (MNIST images are grayscale)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # Output: 16 x 14 x 14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Output: 32 x 7 x 7
        self.conv3 = nn.Conv2d(32, 64, kernel_size=7)  # Output: 64 x 1 x 1, no padding to reduce to a smaller size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Start decoding from a small spatial dimension
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=7)  # Output: 32 x 7 x 7
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: 16 x 14 x 14
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: 1 x 28 x 28, restoring to original size

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Use sigmoid to ensure output values are between 0 and 1
        return x

class DCAE(nn.Module):
    def __init__(self):
        super(DCAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 实例化模型
model = DCAE()

# 打印模型结构
print(model)


# 实例化模型、定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 初始化聚类中心
# TODO：需要初始化聚类中心
cluster_centers = init_cluster_centers(model.encoder, trainloader, n_clusters=10)
# 使用聚类损失训练模型

# 训练模型，整合聚类损失
def train(model, trainloader, optimizer, cluster_centers, epochs=10, lambda_cluster=1.0):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            reconstruction_loss = criterion(outputs, inputs)
            features = model.encoder(inputs).view(inputs.size(0), -1)  # 获取特征表示
            dist = torch.cdist(features, cluster_centers)  # 计算到聚类中心的距离
            clustering_loss = torch.min(dist, dim=1)[0].mean()  # 选择最近的聚类中心，计算平均距离作为聚类损失
            total_loss = reconstruction_loss + lambda_cluster * clustering_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')


# 调用训练函数
train(model, trainloader, optimizer, cluster_centers)






def visualize_clusters(encoder, data_loader, cluster_centers, n_clusters=10, n_images=5):
    features_list = []
    images_list = []
    labels_list = []

    # 使用编码器获取特征和对应的图像
    with torch.no_grad():
        for images, labels in data_loader:
            features = encoder(images).view(images.size(0), -1)
            features_list.append(features)
            images_list.append(images)
            labels_list.append(labels)

    features = torch.cat(features_list, 0)
    images = torch.cat(images_list, 0)
    labels = torch.cat(labels_list, 0)

    # 计算到各个聚类中心的距离
    dist = torch.cdist(features, cluster_centers)

    # 对每个聚类中心，找到最近的n_images个图像
    for i in range(n_clusters):
        cluster_images = images[torch.topk(dist[:, i], largest=False, k=n_images)[1]]

        # 绘制聚类中心的图像
        fig, axs = plt.subplots(1, n_images, figsize=(n_images * 2, 2))
        fig.suptitle(f'Cluster {i + 1}', fontsize=16)
        for j, img in enumerate(cluster_images):
            axs[j].imshow(img.squeeze(), cmap='gray')
            axs[j].axis('off')
        plt.show()


# 使用测试集进行可视化
visualize_clusters(model.encoder, testloader, cluster_centers, n_clusters=10, n_images=5)

