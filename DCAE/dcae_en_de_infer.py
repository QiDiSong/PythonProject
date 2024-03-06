from release_model import DCAE
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random


# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = DCAE()
net.load_state_dict(torch.load("./dcae_encoder_best.pth", map_location=device))
net.to(device)
print(net.parameters())

# MNIST数据集的预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    # transforms.Lambda(lambda x: x.view(-1))
])
# 加载MNIST数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)



# 选择部分数据进行测试和可视化
dataiter = iter(testloader)
images, labels = next(dataiter)
images = images.to(device)
# labels = labels.to(device)
# 选择一个随机索引
random_index = random.randint(0, images.size(0) - 1)
print("random index: ", random_index)
images_random_index = images[random_index].view(-1, 1, 784)  # 调整形状以匹配模型输入
outputs_random_index = net(images_random_index)



# 可视化随机选择的图像和其重建结果
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
plt.imshow(images_random_index.view(28, 28).cpu().detach().numpy(), cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(outputs_random_index.view(28, 28).cpu().detach().numpy(), cmap='gray')
plt.title('Reconstructed Image')
plt.show()